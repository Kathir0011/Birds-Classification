import torch
import torchvision
import os
import gradio as gr
from model import create_vit_b16

# loading the classnames
class_names = []
with open("class_names.txt", "r") as f:
  for cls in f.readlines():
    class_names.append(cls[:-1])

# creating a ViT model
vit_b16_model, vit_transforms = create_vit_b16(num_classes=len(class_names))

vit_b16_model.load_state_dict(
    torch.load(
        f="ViT_B16_510_classes.pth",
        map_location=torch.device("cpu")
    )
)

# creating a predict function
def predict(img):
  """
  Makes prediction on the given image
  """

  img = vit_transforms(img).unsqueeze(dim=0)

  vit_b16_model.eval()
  with torch.inference_mode():
    pred_logits = vit_b16_model(img)
    preds = torch.softmax(pred_logits, dim=1)
    if preds[0].max().item() * 100 > 20:
      # Create a prediction label and prediction probability dictionary for each prediction class
      pred_and_prob_labels = {class_names[i]: preds[0][i].item() for i in range(len(class_names))}

    else:
      pred_and_prob_labels = {"Low Accuracy Warning !!! Kindly verify whether the given image is a Bird.": preds[0].max().item()}

  return pred_and_prob_labels

# creating title, description for the webpage
title = "Birds Classifier 🪶"
description = "Classifies an Image of a Bird to any one of the [510 species](https://huggingface.co/spaces/Kathir0011/Birds_Classification/blob/main/class_names.txt)."
article = "Other Projects:\n"\
"💰 [US Health Insurance Cost Prediction](http://health-insurance-cost-predictor-k19.streamlit.app/)\n"\
"📰 [Fake News Detector](https://fake-news-detector-k19.streamlit.app/)"
# creating examples list
examples_list = [["examples/" + img] for img in os.listdir("examples")]

# Building a gradio app
bird_classification = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    examples = examples_list,
    title=title,
    description=description,
    article=article
)

# launching the web app
bird_classification.launch()
