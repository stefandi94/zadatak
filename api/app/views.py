import json
import os.path as osp

import numpy as np
from PIL import Image
from django.http import HttpResponse
from rest_framework.views import APIView

from app.ml.model import CNNModel
from api.settings import MODEL_WEIGHTS_DIR


class ImagePredictor(APIView):

    def post(self, request, form=None):
        data = request.FILES["data"].file
        image = Image.open(data)

        X = np.array(image, dtype=float)
        X = np.expand_dims(X, axis=0)

        CNN = CNNModel()
        CNN.load_model(osp.join(MODEL_WEIGHTS_DIR, '546-0.135-0.957-0.297-0.945.hdf5'))
        prediction = CNN.predict(X)

        results = {'class': int(prediction[0][0]), 'confidence': float(prediction[0][1])}

        return HttpResponse(json.dumps(results))
