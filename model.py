import tensorflow as tf
import matplotlib.pyplot as plt

N = 5
ITER = 5

# Building the Model
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28,28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
])

logits = model(x_train[:N])
probabilities = tf.nn.softmax(logits)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:N], logits).numpy()

# Training the Model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=ITER)
model.evaluate(x_test, y_test, verbose=2)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

def show_probabilities(images, model, N=5):
    """
    images: batch of images to predict
    model: probability_model (outputs softmax probabilities)
    N: number of images to show
    """
    probs = model(images[:N]).numpy()
    for i, prob in enumerate(probs):
        print(f"Image {i+1} -> Probabilities:")
        for digit, p in enumerate(prob):
            print(f"  Class {digit}: {p:.4f}")
        predicted = prob.argmax()
        confidence = prob.max()
        print(f"Predicted class: {predicted} with probability {confidence:.4f}\n")

        plt.figure(figsize=(6,3))

        plt.subplot(1,2,1)
        plt.imshow(images[i], cmap="gray")
        plt.axis("off")
        plt.title("Input Image")
        
        plt.subplot(1,2,2)
        plt.bar(range(10), prob)
        plt.xticks(range(10))
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.title(f"Predicted: {predicted} ({confidence:.2f})")

        plt.tight_layout()
        plt.show()

show_probabilities(x_test, probability_model, N=5)
