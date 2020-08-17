import streamlit as st
from mantisshrimp.all import *

WEIGHTS_URL = "https://github.com/airctic/streamlitshrimp/releases/download/pets_faster_resnetfpn50/pets_faster_resnetfpn50.zip"
IMAGE_URL = "https://petcaramelo.com/wp-content/uploads/2018/06/beagle-cachorro.jpg"
IMG_PATH = "tmp.jpg"

st.title("MantisShrimp Demo App")

class_map = datasets.pets.class_map()

model = faster_rcnn.model(num_classes=len(class_map))
state_dict = torch.hub.load_state_dict_from_url(
    WEIGHTS_URL, map_location=torch.device("cpu")
)
model.load_state_dict(state_dict)

download_url(IMAGE_URL, IMG_PATH)
img = open_img(IMG_PATH)
tfms = tfms.A.Adapter([tfms.A.Normalize()])
# Whenever you have images in memory (numpy arrays) you can use `Dataset.from_images`
infer_ds = Dataset.from_images([img], tfms)

batch, samples = faster_rcnn.build_infer_batch(infer_ds)
preds = faster_rcnn.predict(model=model, batch=batch)
# Show preds by grabbing the images from `samples`
imgs = [sample["img"] for sample in samples]
show_preds(
    imgs=imgs,
    preds=preds,
    class_map=class_map,
    denormalize_fn=denormalize_imagenet,
    show=True,
)

fig = plt.gcf()
fig.canvas.draw()
fig_arr = np.array(fig.canvas.renderer.buffer_rgba())

st.image(fig_arr)