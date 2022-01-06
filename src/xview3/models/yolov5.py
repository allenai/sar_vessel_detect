import numpy
import os.path
import sys
import torch
import yaml

class ImportContext(object):
	def __init__(self):
		self.expected_path = os.path.join('.', 'yolov5')

	def __enter__(self):
		# from github.com/ultralytics/yolov5
		sys.path.insert(1, self.expected_path)
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		# reset sys.modules
		for module_name in list(sys.modules.keys()):
			if not hasattr(sys.modules[module_name], '__file__'):
				continue
			fname = sys.modules[module_name].__file__
			if fname is None:
				continue
			if not fname.startswith(self.expected_path):
				continue
			del sys.modules[module_name]
		sys.path.remove(self.expected_path)

def create_model(num_classes, num_channels, device, config, image_size=800):
	pretrained = config.getboolean("Pretrained", fallback=True)
	backbone = config.get("Backbone", fallback="x")

	with ImportContext() as ctx:
		import utils.general
		import utils.loss
		import utils.torch_utils
		import models.yolo

		class Yolov5(torch.nn.Module):
			def __init__(self, device, nc=3, mode='x'):
				super(Yolov5, self).__init__()
				self.nc = nc

				# e.g. 's', 'm', 'l', 'x'
				self.mode = mode

				self.confidence_threshold = 0.05
				self.iou_threshold = 0.5

				print('yolov5: set nc={}, mode={}, conf={}, iou={}'.format(self.nc, self.mode, self.confidence_threshold, self.iou_threshold))

				with open(os.path.join(ctx.expected_path, 'data', 'hyps', 'hyp.scratch.yaml'), 'r') as f:
					hyp = yaml.load(f, Loader=yaml.FullLoader)

				self.model = models.yolo.Model(cfg=os.path.join(ctx.expected_path, 'models', 'yolov5{}.yaml'.format(self.mode)), nc=self.nc, anchors=hyp.get('anchors'))
				# we need to set it onto device early since compute_loss copies the anchors
				self.model.to(device)

				self.ema = utils.torch_utils.ModelEMA(self.model)

				nl = self.model.model[-1].nl

				hyp['box'] *= 3. / nl
				hyp['cls'] *= self.nc / 80. * 3. / nl
				hyp['obj'] *= (image_size / 640) ** 2 * 3. / nl
				print(hyp)

				self.model.nc = self.nc
				self.model.hyp = hyp
				self.model.gr = 1.0
				self.model.names = ['object{}'.format(i) for i in range(self.nc)]

				weights = numpy.zeros((self.nc,), dtype='float32')
				weights[0:3] = 0.33
				weights[3:] = 1e-3
				self.model.class_weights = torch.from_numpy(weights)

				self.compute_loss = utils.loss.ComputeLoss(self.model)

				self.mloss = []

			def forward(self, x, targets=None):
				x = torch.stack(x, dim=0)
				device = x.device

				if self.training:
					# Convert targets to YOLOv5 format.
					boxes = []
					counts = []
					cls_labels = []

					for target_dict in targets:
						# boxes: xyxy -> xywh
						boxes.append(torch.stack([
							(target_dict['boxes'][:, 0] + target_dict['boxes'][:, 2]).float() / 2 / x.shape[3],
							(target_dict['boxes'][:, 1] + target_dict['boxes'][:, 3]).float() / 2 / x.shape[2],
							(target_dict['boxes'][:, 2] - target_dict['boxes'][:, 0]).float() / x.shape[3],
							(target_dict['boxes'][:, 3] - target_dict['boxes'][:, 1]).float() / x.shape[2],
						], dim=1))
						counts.append(target_dict['boxes'].shape[0])
						if target_dict['boxes'].shape[0] == 0:
							cls_labels.append(torch.zeros((0,), dtype=torch.int64, device=device))
						else:
							cls_labels.append(target_dict['labels']-1)

					boxes = torch.cat(boxes, dim=0)
					counts = torch.tensor(counts, dtype=torch.int32, device=device)
					cls_labels = torch.cat(cls_labels, dim=0)

					# output: list of detections with:
					# - first column indicating image index
					# - second column indicating class index
					indices = torch.repeat_interleave(
						torch.arange(counts.shape[0], dtype=torch.int32, device=counts.device).float(),
						counts.long()
					).reshape(-1, 1)
					targets = torch.cat([indices, cls_labels.reshape(-1, 1), boxes], dim=1)

					pred = self.model(x)
					loss, loss_info = self.compute_loss(pred, targets)

					self.mloss.append(loss_info.cpu().detach().numpy())
					if len(self.mloss) > 256:
						print([numpy.mean([t[i] for t in self.mloss]) for i in range(len(self.mloss[0]))])
						del self.mloss[:]

					return {'loss': torch.mean(loss)}
				else:
					inf_out, _ = self.model(x)
					#inf_out, _ = self.ema.ema(x)
					detections = utils.general.non_max_suppression(inf_out, conf_thres=self.confidence_threshold, iou_thres=self.iou_threshold)
					# yolov5 returns a list of lists [xyxy, conf, cls] (one sub-list per image)
					outputs = []
					for dlist in detections:
						boxes = dlist[:, 0:4]
						scores = dlist[:, 4]
						labels = dlist[:, 5]+1
						outputs.append({
							'boxes': boxes,
							'scores': scores,
							'labels': labels,
						})
					return outputs

		model = Yolov5(device, nc=num_classes, mode=backbone)

		if pretrained:
			pretrained_path = os.path.join(ctx.expected_path, 'yolov5{}.pt'.format(backbone))
			print('yolov5: load pretrained model from {}'.format(pretrained_path))
			state_dict = torch.load(pretrained_path)['model'].state_dict()
			state_dict = {'model.'+k: v for k, v in state_dict.items()}

			# Compare against model.state_dict and remove things with mismatching size.
			my_state_dict = model.state_dict()
			filter_state_dict = {k: v for k, v in state_dict.items() if k in my_state_dict and my_state_dict[k].shape == v.shape}

			missing_keys, unexpected_keys = model.load_state_dict(filter_state_dict, strict=False)
			print('... of {} keys in pre-trained dict, got {} missing and {} unexpected/mismatched'.format(len(state_dict), len(missing_keys), len(state_dict) - len(filter_state_dict)))

	return model
