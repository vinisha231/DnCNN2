import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Subtract
from tensorflow.keras.models import Model
import numpy as np
import os

class DnCNN:
    def __init__(self, sess=None, batch_size=128):
        self.sess = sess
        self.batch_size = batch_size
        self.build_model()

    def build_model(self):
        input_shape = (None, None, 1)  # Variable input size, single channel (grayscale)
        inputs = Input(shape=input_shape)
        
        # Convolutional layers
        x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
        for _ in range(18):
            x = self.conv_block(x)
        
        # Final convolutional layer
        outputs = Conv2D(filters=1, kernel_size=3, padding='same')(x)
        
        # Subtracting the noisy image from the denoised image
        outputs = Subtract()([inputs, outputs])
        
       	self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def conv_block(self, x):
        x = Conv2D(filters=64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def train(self, ndct_data, ldct_data, eval_data, ldct_eval_data, batch_size=128, ckpt_dir='./checkpoint', epoch=50, lr=0.001, sample_dir='./sample'):
        # Assuming ndct_data, ldct_data, eval_data, ldct_eval_data are TensorFlow tensors or placeholders
        num_batches = tf.data.experimental.cardinality(ndct_data).numpy() // batch_size

        # Your training loop here
        for epoch in range(epoch):
            for batch_idx in range(num_batches):
                # Fetch or generate batch data
                ndct_batch = ndct_data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                ldct_batch = ldct_data[batch_idx * batch_size:(batch_idx + 1) * batch_size]

                # Training steps
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.images: ndct_batch, self.labels: ldct_batch})
                
                # Print or log loss
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}/{epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}")

            # Save checkpoint or sample images
            if epoch % 10 == 0:
                self.save(ckpt_dir, epoch)
                self.sample(sample_dir)

        # Evaluate model after training
        self.evaluate(eval_data, ldct_eval_data)

    def test(self, test_data, ckpt_dir, save_dir):
        # Load the latest checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        if latest_checkpoint:
            self.model.load_weights(latest_checkpoint)
            print(f"Loaded checkpoint {latest_checkpoint}")
        else:
            print("No checkpoint found.")
            return
        
        # Perform testing
        # Example: Test on the test_data
        denoised_images = self.model.predict(test_data)
        
        # Save or use the results as needed
        save_path = os.path.join(save_dir, "denoised_images.npy")
        np.save(save_path, denoised_images)
        print(f"Denoised images saved at {save_path}")

