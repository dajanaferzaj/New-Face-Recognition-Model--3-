import cv2 as cv
import pandas as pd
import numpy as np


detector = cv.FaceDetectorYN.create(
    'face_detection_yunet_2023mar.onnx',
    "",
    (320, 320),
    0.7,
    0.3,
    5000,
    backend_id=3
)
recognizer = cv.FaceRecognizerSF.create('face_recognition_sface_2021dec.onnx',"", backend_id=3)
cosine_similarity_threshold = 0.35

embeddings_df = pd.read_csv("embeddings_df.csv")
embeddings_df['Embedding'] = embeddings_df['Embedding'].apply(lambda x: np.array([float(t) for t in x[2:-2].split()]))


def visualize(input, faces, identities = [], fps = 0, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            if identities:
                cv.putText(input, identities[idx], (coords[0], coords[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


cap = cv.VideoCapture(0)

new_resolution = 1

while(True):
    ret, frame = cap.read()
    if ret:
        pic_width = int(frame.shape[1] * new_resolution)
        pic_height = int(frame.shape[0] * new_resolution)
        new_dimension = (pic_width, pic_height)

        frame = cv.resize(frame, new_dimension, interpolation=cv.INTER_AREA)


        detector.setInputSize([frame.shape[1], frame.shape[0]])
        faces = detector.detect(frame)

        identities = []
        if faces[1] is not  None:
            for i in range(len(faces[1])):
                coords = faces[1][i].astype(np.int32)
                if (faces[1][i][-1] >= 0.7):
                    face1_align = recognizer.alignCrop(frame, faces[1][i])
                    face1_feature = recognizer.feature(face1_align)
                    face1_feature = recognizer.feature(face1_align)
                    identity = ""
                    embeddings_df['cosine'] = embeddings_df.Embedding.apply(
                                                    lambda x: int(recognizer.match(face1_feature, np.reshape(x,(1,-1)).astype(np.float32), cv.FaceRecognizerSF_FR_COSINE) > cosine_similarity_threshold))
                    a = embeddings_df[embeddings_df.cosine == 1]
                    if len(a):
                        identity = a.Name.iloc[0].capitalize()

                    if not identity:
                        identity = "Unknown"

                    identities.append(identity)
        
        visualize(frame, faces, identities)

        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            break

#%%
