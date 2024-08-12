import os
import torch
import torch_neuron
from model import GRIPImageModelForMultiOuputClassification


class GRIPImageModelScript(torch.nn.Module):
    def __init__(self, model, num_hint):
        super().__init__()
        self.vit = model.image_embedding_model
        self.category_classifier = model.category_classifier
        self.color_classifier = model.color_classifier
        self.num_hint = num_hint

    def forward(self, num_hint, pixel_values):
        outputs = self.vit(pixel_values)
        last_hidden_state = outputs[0]
        sequence_output = last_hidden_state[:, 0, :]

        # One-Hot
        hint_oh = torch.nn.functional.one_hot(hint, num_classes=self.num_hint).view(-1, self.num_hint)

        # Concatenate
        sequence_output = torch.concat([sequence_output, hint_oh], axis=1)

        # Classifier Head(Logits)
        category_logits = self.category_classifier(sequence_output)
        color_logits = self.color_classifier(sequence_output)
        return category_logits, color_logits



# input sample
hint = torch.LongTensor([[0], [1]])
pixel_values = torch.rand([2, 3, 224, 224])
trained_model = GRIPImageModelForMultiOuputClassification.load_from_checkpoint(checkpoint)
model = GRIPImageModelScript(trained_model, num_hint)

neurn_model = torch.neuron.trace(
    model, [hint, pixel_values], dynamic_batch_size=True
)

neurn_model.save("model.pt")