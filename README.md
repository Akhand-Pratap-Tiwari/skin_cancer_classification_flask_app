# Flask App for Skin Cancer classification using IRV2 and Soft Attention

A flask app to run cancer classifier and classifiy cancer into 7 given classes:
- `akiec`: Actinic Keratoses and Intraepithelial Carcinoma/Bowen disease
- `bcc`: Basal Cell Carcinoma
- `bkl`: Benign lesions of the Keratosis type (solar lentigine/seborrheic keratoses and lichen-planus like keratosis)
- `df`: Dermatofibroma
- `mel`: Melanoma
- `nv`: Melanocytic Nevi
- `vasc`: Vascular Lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhages)

File descriptions:
- The `app.py` contains the main logic for the flask app.
- The `example_image.jpg` is the exmaple image which is sent in encoded form in base64 format.
- `example_request.ipynb` shows how to use this flask app.
- `requirements.txt` contains the main requirements for the project.
- `saved_model_v4.hdf5` is the actual model being used for the classification.

## How to run:
- First install Python 3.11.6.
- Install everything mentioned in requirements.txt.
- Install any additional libs if any error occurs.
- Now, in the root directroy run the `flask run` command.
- Now, in the run the jupyter notebook to see the response as output.

This project was made more as microservice rather than being a standalone service so it might not meet your taste but it demonstrates the required functionality. You can see the example notebook for more info.   

## More to read
If you want to dive deeper into the model code and soft attention then have a look at this:
- [paper 1](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7445643/)
- [paper 2](https://arxiv.org/pdf/2105.03358v3) along with code [here](https://github.com/skrantidatta/Attention-based-Skin-Cancer-Classification)
- The dataset used in this model was a balanced subset of HAM10000 dataset.
