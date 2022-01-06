import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        def down_layer(in_channels, out_channels):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                torch.nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03),
                torch.nn.ReLU(inplace=True),
                #torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
                #torch.nn.ReLU(inplace=True),
            )

        self.layer1 = down_layer(3, 32) # -> 64x64
        self.layer2 = down_layer(32, 64) # -> 32x32
        self.layer3 = down_layer(64, 128) # -> 16x16
        self.layer4 = down_layer(128, 256) # -> 8x8
        self.layer5 = down_layer(256, 512) # -> 4x4
        self.layer6 = down_layer(512, 512) # -> 2x2

        self.pred_length = torch.nn.Conv2d(512, 1, 4, stride=2, padding=1)
        self.pred_confidence = torch.nn.Conv2d(512, 3, 4, stride=2, padding=1)
        self.pred_correct = torch.nn.Conv2d(512, 2, 4, stride=2, padding=1)
        self.pred_source = torch.nn.Conv2d(512, 3, 4, stride=2, padding=1)
        self.pred_fishing = torch.nn.Conv2d(512, 2, 4, stride=2, padding=1)
        self.pred_vessel = torch.nn.Conv2d(512, 2, 4, stride=2, padding=1)

    def forward(self, x, targets=None):
        device = x.device
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)

        length_scores = self.pred_length(layer6)[:, 0, 0, 0]
        confidence_scores = self.pred_confidence(layer6)[:, :, 0, 0]
        correct_scores = self.pred_correct(layer6)[:, :, 0, 0]
        source_scores = self.pred_source(layer6)[:, :, 0, 0]
        fishing_scores = self.pred_fishing(layer6)[:, :, 0, 0]
        vessel_scores = self.pred_vessel(layer6)[:, :, 0, 0]

        if self.training:
            # Length.
            length_labels = targets['vessel_length']
            valid_length_indices = length_labels >= 0
            valid_length_labels = length_labels[valid_length_indices]
            valid_length_scores = length_scores[valid_length_indices]
            if len(valid_length_labels) > 0:
                length_loss = torch.div(torch.abs(valid_length_labels - valid_length_scores), valid_length_labels).mean()
            else:
                length_loss = torch.zeros((1,), dtype=torch.float32, device=device)

            def get_ce_loss(labels, scores):
                valid_indices = labels >= 0
                valid_labels = labels[valid_indices]
                valid_scores = scores[valid_indices, :]
                if len(valid_labels) > 0:
                    return torch.nn.functional.cross_entropy(valid_scores, valid_labels)
                else:
                    return torch.zeros((1,), dtype=torch.float32, device=device)

            confidence_loss = get_ce_loss(targets['confidence'], confidence_scores)
            correct_loss = get_ce_loss(targets['correct'], correct_scores)
            source_loss = get_ce_loss(targets['source'], source_scores)
            fishing_loss = get_ce_loss(targets['fishing'], fishing_scores)
            vessel_loss = get_ce_loss(targets['vessel'], vessel_scores)

            loss = length_loss + confidence_loss + correct_loss + source_loss + fishing_loss + vessel_loss

            return {
                'loss': loss,
                'length': length_loss,
                'confidence': confidence_loss,
                'correct': correct_loss,
                'source': source_loss,
                'fishing': fishing_loss,
                'vessel': vessel_loss,
            }
        else:
            return (
                length_scores,
                torch.softmax(confidence_scores, dim=1),
                torch.softmax(correct_scores, dim=1),
                torch.softmax(source_scores, dim=1),
                torch.softmax(fishing_scores, dim=1),
                torch.softmax(vessel_scores, dim=1),
            )
