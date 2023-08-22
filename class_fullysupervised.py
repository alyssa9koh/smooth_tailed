
# Dataset that sets up getting items for the pretraining step.
class PretrainDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        # Normalization from Sklearn
        self.normalizer = StandardScaler()

    def __len__(self):
        return len(self.img_labels)

    # This function returns three images: the anchor, the positive, and the negative.
    # The anchor image is the main image.
    # The positive image is the anchor image, but augmented.
    # The negative image is a completely different image. This image may be of the same species as the anchor.
    def __getitem__(self, idx):
        # print("---------------- GET ITEM START -------------")
        timestart = time.time()

        time9 = time.time()
        # setting image path
        num = self.img_labels.iloc[idx, 13] # should correspond to "photo_id"
        num = str(num)
        img_path = os.path.join(self.img_dir, num[:3], num) + ".png"
        # print(img_path)
        time10 = time.time()
        # print("LABEL GET: ", time10 - time9)

# new_path = os.path.join(train_annotations_path, "123", "1234567") + ".png"

        time1 = time.time()
        # getting anchor image
        anchor_image = Image.open(img_path) # Reading image
        anchor_label = self.img_labels.iloc[idx, 10] # Getting label
        # anchor_image = self.normalizer.transform(anchor_image.reshape(1, -1)).reshape(anchor_image.shape) # Normalize the data within __getitem__
        time2 = time.time()
        # print("ANCHOR IMAGE GET: ", time2 - time1)

        time5 = time.time()
        # getting our negative image, aka another random unique image
        negative_indices = np.where(np.arange(len(self.img_labels)) != idx)[0]
        negative_index = np.random.choice(negative_indices)
        negative_num = self.img_labels.iloc[negative_index, 13]
        negative_num = str(negative_num)
        negative_img_path = os.path.join(self.img_dir, negative_num[:3], negative_num) + ".png"
        negative_image = Image.open(negative_img_path)
        time6 = time.time()
        # print("NEGATIVE IMAGE GET: ", time6 - time5)

        if self.transform: # If a self transform function is specified, apply it
            time7 = time.time()
            anchor_image = self.transform(anchor_image)
            # positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)
            time8 = time.time()
            # print("TRANSFORM ALL IMAGES: ", time8 - time7)

        time3 = time.time()
        # augmenting anchor image to create our positive
        positive_image = None # initializing it as none so that we can transform it later
        positive_image = self.target_transform(anchor_image)
        time4 = time.time()
        # print("POSITIVE IMAGE GET: ", time4 - time3)

        timeend = time.time()
        # print("GETITEM: ", timeend - timestart)
        return anchor_image, positive_image, negative_image