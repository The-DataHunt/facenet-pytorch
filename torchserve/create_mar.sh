torch-model-archiver \
--model-name facenet \
--version 1.0 \
--model-file ../models/FaceDetector.py \
--serialized-file ../data/onet.pt \
--extra-files ../models/mtcnn.py,../models/utils/detect_face.py,../data/pnet.pt,../data/rnet.pt \
--handler FacenetHandler:handle \
--export-path /home/ubuntu/work/InferenceModels/model_store \
-f