import os
import numpy as np
import pandas as pd
import cv2
import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import time
from sklearn.decomposition import PCA

# Shift and Scale Data
class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)

        self.std = np.std(X, axis=0, ddof=1)
        self.std[self.std == 0] = 1.0

        return (X - self.mean) / self.std

    def transform(self, X):
        return (X - self.mean) / self.std

class Emotion_Wrapper:
    def __init__(self, model_path='emotion_classifier_model.pkl'):

        # HOG parameters
        self.hog_params = {'orientations': 12, 'pixels_per_cell': (6,6), 
                           'cells_per_block': (2,2), 'block_norm': 'L1-sqrt'}
        
        self.pca_components = 900 # Reduced dimensionality
        self.model_path = model_path

        # Model components
        self.scaler, self.pca, self.classifier = None, None, None
        self.emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

    # Pre-processing
    def augment_image(self, image):
        aug = [image] 
        
        # Flip Image
        aug.append(cv2.flip(image, 1))

        # Rotation
        angle = np.random.uniform(-15, 15)
        (h, w) = image.shape
        M_rot = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        aug.append(cv2.warpAffine(image, M_rot, (w, h), borderMode=cv2.BORDER_REPLICATE))

        # Translation
        tx, ty = np.random.randint(-3, 4), np.random.randint(-3, 4)
        M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        aug.append(cv2.warpAffine(image, M_trans, (w, h), borderMode=cv2.BORDER_REPLICATE))

        # Add noise
        noise = np.random.normal(0, 2, image.shape).astype(np.uint8)
        aug.append(cv2.add(image, noise))

        # Lighting variation
        gamma = np.random.uniform(0.8, 1.2)
        
        table = []
        for i in range(256):
            val = ((i / 255.0) ** (1.0 / gamma)) * 255
            table.append(val)
        table = np.array(table, dtype="uint8")

        aug.append(cv2.LUT(image, table))

        return aug

    # Average HOG Extraction w/ CLAHE
    def extract_features(self, image):
        hog_ = hog(image, **self.hog_params)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        hog_clahe = hog(clahe.apply(image), **self.hog_params)

        return (hog_ + hog_clahe) / 2.0

    # Load Data
    def load_data_from_csvs(self, fer_path, fer_plus_path, use="Training", augment=True):
        fer_data = pd.read_csv(fer_path)
        fer_plus_data = pd.read_csv(fer_plus_path)
        fer_plus_data.columns = fer_plus_data.columns.str.lower()

        feats, labs, count = [], [], 0

        for idx in range(len(fer_data)):
            if use.lower() not in fer_data.loc[idx, 'Usage'].strip().lower():
                continue

            pixels = np.array([int(p) for p in fer_data.loc[idx, 'pixels'].split()], dtype=np.uint8)

            if pixels.size != 48*48: 
                continue

            image = pixels.reshape((48, 48))

            votes = [int(fer_plus_data.loc[idx, key]) for key in self.emotions]

            label = int(np.argmax(votes))

            for img in (self.augment_image(image) if augment else [image]):
                feats.append(self.extract_features(img))
                labs.append(label)
                count += 1

        print(f"Got {count} samples for {use}.")
        return np.array(feats), np.array(labs)

    def fit(self, fer_path, fer_plus_path):
        # Load training data
        print("Loading Training Data...")
        X, y = self.load_data_from_csvs(fer_path, fer_plus_path)
        if X.size == 0: 
            raise ValueError("Loaded 0 samples.")
        
        # Scale data
        print("Scaling Data...")
        self.scaler = Scaler() # StandardScaler()
        X_scaled = self.scaler.fit(X)
        comp = min(self.pca_components, X_scaled.shape[1])
        if comp != self.pca_components:
            print(f"Adjusting PCA components to {comp}.")
            self.pca_components = comp
        print('Scaling Done.')

        # Principal component analysis
        print("Applying PCA...")
        self.pca = PCA(n_components=self.pca_components)
        X_pca = self.pca.fit_transform(X_scaled)
        var = np.sum(self.pca.explained_variance_ratio_)
        print(f"PCA reduced from {X_scaled.shape[1]} to {X_pca.shape[1]} dimensions, maintained {var*100:.2f}% variance.")

        # Train classifier
        print('Training Classifier...')
        start = time.time()
        self.classifier = LinearSVC(C=0.01, dual=False, max_iter=10000) # C = 0.1 ??
        self.classifier.fit(X_pca, y)
        
        # Save model
        joblib.dump((self.classifier, self.pca, self.scaler), self.model_path)
        print(f"Training done in {time.time() - start:.2f}s. Model saved to {self.model_path}.\n")

    def test(self, fer_path, fer_plus_path, use="PublicTest", test_time_augmentation=True):
        print(f"Testing model on {use}...")
        print(f"Test Time Augmentation: {test_time_augmentation}")

        # Load model if not trained
        if self.classifier is None or self.pca is None or self.scaler is None:
            self.classifier, self.pca, self.scaler = joblib.load(self.model_path)
            print(f"Loaded model from {self.model_path}.")

        # Test with or without test-time augmentation
        X, y = self.load_data_from_csvs(fer_path, fer_plus_path, use=use, augment=test_time_augmentation)

        if test_time_augmentation:
            num_aug = 6  # Augmented 5 times and original

            if X.shape[0] % num_aug != 0: 
                print(f"Warning: Augmented count not a multiple of {num_aug}.")

            y = y[::num_aug]
            X_scaled = self.scaler.transform(X)
            X_pca = self.pca.transform(X_scaled)

            preds = self.classifier.predict(X_pca).reshape((X.shape[0] // num_aug), num_aug)
            preds = mode(preds, axis=1, keepdims=True)[0].flatten()  # Majority decision
        else:
            if X.shape[0] == 0:
                raise ValueError("No testing samples found.")

            X_scaled = self.scaler.transform(X)
            X_pca = self.pca.transform(X_scaled)

            preds = self.classifier.predict(X_pca)
            
        acc = accuracy_score(y, preds)
        print(f'Testing Accuracy: {acc*100:.2f}%\n')

    def predict(self, image=None, img_path=None):
        start = time.time()
        if image is None and img_path is None:
            raise ValueError("Either image or img_path must be provided.")

        # Load model if not trained
        if self.classifier is None or self.pca is None or self.scaler is None:
            self.classifier, self.pca, self.scaler = joblib.load(self.model_path)
            print(f"Loaded model from {self.model_path}.")

        output_paths = []

        if img_path is not None:
            for path in img_path:
                original_img = cv2.imread(path)
                if original_img is None: 
                    print(f"Error reading {path}.")
                    continue
                    
                # Grayscale
                img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                if img.shape != (48,48): 
                    img = cv2.resize(img, (48,48))

                # Predict
                imgs = self.augment_image(img)
                feats = [self.extract_features(i) for i in imgs]
                feats_pca = self.pca.transform(self.scaler.transform(feats))
                pred = mode(self.classifier.predict(feats_pca), keepdims=True)[0][0]

                # Label
                label = self.emotions[int(pred)]
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = original_img.shape[1] / 1000
                thickness = max(1, int(scale * 2))
                margin = int(15 * scale)

                (text_w, text_h), _ = cv2.getTextSize(label, font, scale, thickness)
                x = original_img.shape[1] - text_w - margin
                y = text_h + margin

                cv2.rectangle(original_img, 
                            (x - margin//2, y - text_h - margin//2),
                            (x + text_w + margin//2, y + margin//2),
                            (245, 245, 245), -1)
                cv2.putText(original_img, label, (x, y), font, scale,
                        (45, 45, 45), thickness, cv2.LINE_AA)

                base, ext = os.path.splitext(path)
                out_path = f"{base}_annotated{ext}"
                cv2.imwrite(out_path, original_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                output_paths.append(out_path)

        if image is not None:
            annotated_img = image.copy()
            
            # Grayscale
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if img.shape != (48,48):
                img = cv2.resize(img, (48,48))

            # Predict
            feats = [self.extract_features(img)]
            feats_pca = self.pca.transform(self.scaler.transform(feats))
            pred = mode(self.classifier.predict(feats_pca), keepdims=True)[0][0]

            # Label
            label = self.emotions[int(pred)]
            scale = image.shape[1] / 1000
            thickness = max(1, int(scale * 2))
            margin = int(15 * scale)
            (text_w, text_h), _ = cv2.getTextSize(label, font, scale, thickness)
            x = image.shape[1] - text_w - margin
            y = text_h + margin

            cv2.rectangle(annotated_img, 
                        (x - margin//2, y - text_h - margin//2),
                        (x + text_w + margin//2, y + margin//2),
                        (245, 245, 245), -1)
            cv2.putText(annotated_img, label, (x, y), font, scale,
                    (45, 45, 45), thickness, cv2.LINE_AA)

            output_paths.append(annotated_img)

        print(f"Emotion: {self.emotions[int(pred)]}\tTime Elapsed: {time.time() - start:.2f}s")
        return output_paths if output_paths else None
    
    def predict_video(self, video_path, output_path):
        # Load model if not trained
        if self.classifier is None or self.pca is None or self.scaler is None:
            self.classifier, self.pca, self.scaler = joblib.load(self.model_path)
            print(f"Loaded model from {self.model_path}.")

        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening {video_path}")
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                            (w, h))
        
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Predict
            processed = self.predict(image=frame)
            if processed is not None:
                annotated_frame = processed[0]
                out.write(annotated_frame)
            else:
                out.write(frame)
                
            cnt += 1
            if cnt % 100 == 0:
                elapsed = time.time() - start_time
                print(f"{cnt}/{total_frames}\t({cnt/elapsed:.2f} FPS)")
        
        cap.release()
        out.release()
        
        print(f"Saved to {output_path}")
        return output_path

    def test_full(self, fer_path, fer_plus_path, test_time_augmentation=True):
        for use in ["Training", "PublicTest", "PrivateTest"]:
            self.test(fer_path, fer_plus_path, use=use, test_time_augmentation=test_time_augmentation)

if __name__ == "__main__":
    fer_path = "fer2013.csv"
    fer_plus_path = "fer2013new.csv"
    
    clf = Emotion_Wrapper()

    # clf.fit(fer_path, fer_plus_path)

    # clf.test_full(fer_path, fer_plus_path, test_time_augmentation=True)

    custom_images = ['sample_imgs/angry.jpg', 'sample_imgs/happy.jpg', 'sample_imgs/sad.jpg', 'sample_imgs/fear.jpg', 'sample_imgs/disgust.jpg', 'sample_imgs/surprise.jpg', 'sample_imgs/neutral.png']  

    clf.predict(img_path=custom_images)
