import os
import pandas as pd
import cv2 as cv
import numpy as np


embeddings_df = pd.DataFrame(columns=['Name', 'Embedding'])

detector = cv.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    0.7,
    0.3,
    5000,
    backend_id=3
)
recognizer = cv.FaceRecognizerSF.create(
    "face_recognition_sface_2021dec.onnx", "", backend_id=3)


path_to_images = "Faces"


for filename in os.listdir(path_to_images):
    img = cv.imread(f"{path_to_images}/{filename}")
    detector.setInputSize([img.shape[1], img.shape[0]])
    face = detector.detect(img)
    assert face[1] is not None, f'Cannot find a face in {filename}'
    coords = face[1][0].astype(np.int32)
    if (coords[2] >= 40) and (coords[3] >= 40):
        face_align = recognizer.alignCrop(img, face[1][0])
        face_feature = recognizer.feature(face_align)
        x = {"Name": filename[:-4], "Embedding": face_feature}
        embeddings_df = pd.concat(
            [
                embeddings_df, 
                pd.DataFrame.from_dict([x], orient='columns')
            ], 
            ignore_index=True)

embeddings_df.to_csv("embeddings_df.csv", index=False)


# for filename in os.listdir("./video_face_images/"):
#     img = cv.imread(cv.samples.findFile(f"./video_face_images/{filename}"))
#     detector.setInputSize([img.shape[1], img.shape[0]])
#     face = detector.detect(img)
#     assert face[1] is not None, f'Cannot find a face in {filename}'
#     coords = face[1][0].astype(np.int32)
#     if (coords[2] >= 40) and (coords[3] >= 40):
#         face_align = recognizer.alignCrop(img, face[1][0])
#         face_feature = recognizer.feature(face_align)
#         x = {"Name": filename[:-4], "Embedding": face_feature}
#         embeddings_df = pd.concat([embeddings_df, pd.DataFrame.from_dict(
#             [x], orient='columns')], ignore_index=True)

# embeddings_df.to_csv("video_embeddings.csv", index=False)

#%%
