from keras.models import load_model  # TensorFlow é necessário para Keras funcionar
from PIL import Image, ImageOps  # Instale pillow em vez de PIL
import numpy as np  # Biblioteca para manipulação de arrays

# Desativar notação científica para clareza
np.set_printoptions(suppress=True)

# Carregar o modelo Keras
model = load_model(r"C:\Users\LEONARDO\Documents\KERAS_REDE_NEURAL\keras_model.h5", compile=False)

# Carregar os rótulos das classes
class_names = open("labels.txt", "r").readlines()

# Criar o array no formato correto para alimentar o modelo Keras
# O 'length' ou número de imagens que você pode colocar no array é
# determinado pela primeira posição na tupla de forma, neste caso 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Substitua isso pelo caminho da sua imagem
image = Image.open(r"C:\Users\LEONARDO\Documents\KERAS_REDE_NEURAL\modeda.jpeg").convert("RGB")

# Redimensionar a imagem para pelo menos 224x224 e depois recortar do centro
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Converter a imagem em um array numpy
image_array = np.asarray(image)

# Normalizar a imagem
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Carregar a imagem no array
data[0] = normalized_image_array

# Fazer a previsão usando o modelo
prediction = model.predict(data)
index = np.argmax(prediction)  # Obter o índice da classe com a maior probabilidade
class_name = class_names[index]  # Obter o nome da classe correspondente ao índice
confidence_score = prediction[0][index]  # Obter a pontuação de confiança da previsão

# Imprimir a previsão e a pontuação de confiança
print("Class:", class_name[2:], end="")  # O [2:] remove possíveis caracteres iniciais indesejados
print("Confidence Score:", confidence_score)
