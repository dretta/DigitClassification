try:
    from tfserve import TFServeApp
    import numpy as np
    import PIL
    import tempfile
except Exception as importErr:
    print("-E- Module import failed")
    raise importErr


class DigitClassificationServer:

    def __init__(self):
        self.inputTensor = "import/input_tensor"
        self.outputTensor = "import/output_tensor/Softmax"
        self.digit_names = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

        # Create the Inference Server and run it
        app = TFServeApp("Mnist.pb", [self.inputTensor], [self.outputTensor], self.encode, self.decode)
        app.run('127.0.0.1', 5000)

    def encode(self, request_data):
        """!
        The encode operation for TfServe

        @param request_data: A byte representation of the image sent to the server
        @return A dictionary connecting the image with the input tensor name
        """
        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            # Write the request image to a temp file
            f.write(request_data)

            # Modify the file to match the required input
            img = PIL.Image.open(f).resize((28, 28)).convert('L')
            img = np.asarray(img) / 255.
            img = img.astype('float32')
            img = img.reshape((img.shape[0], img.shape[1], 1))

        return {self.inputTensor: img}

    def decode(self, outputs):
        """!
        The decode operation for TfServe

        @param outputs: A dictionary connecting the output tensor with the result category
        @return A dictionary of data that is returned to the client
        """
        p = outputs[self.outputTensor + ":0"]
        index = np.argmax(p)
        return {"class": self.digit_names[index], "prob": float(p[index])}


if __name__ == '__main__':
    DigitClassificationServer()
