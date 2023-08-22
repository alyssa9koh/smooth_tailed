
class OracleDataset(Dataset):
      def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        # Normalization from Sklearn
        self.normalizer = StandardScaler()

      def __len__(self):
        return len(self.img_labels)

      def __getitem__(self, idx):
        # getting anchor image
        # setting image path
        num = self.img_labels.iloc[idx, 13] # gathers "photo_id"
        num = str(num)
        anchor_img_path = os.path.join(self.img_dir, num[:3], num) + ".png" # Getting file path of image
        anchor_image = Image.open(anchor_img_path) # Reading image
        anchor_label = self.img_labels.iloc[idx, 4] # Getting label; this is the taxon_id. Species name is index number 10

        # select positive sample with the same label as anchor
        # WE NEED to make sure that the positive image IS NOT the same as the anchor,
        # BUT shares the same species id as the anchor.
        positive_indices = np.where(self.img_labels['taxon_id'] == anchor_label)[0]
        positive_index = np.random.choice(positive_indices)
        positive_num = self.img_labels.iloc[positive_index, 13]
        positive_num = str(positive_num)
        positive_img_path = os.path.join(self.img_dir, positive_num[:3], positive_num) + ".png"
        positive_image = Image.open(positive_img_path)

        # getting our negative image, aka another random unique image
        negative_indices = np.where(self.img_labels['taxon_id'] != anchor_label)[0]
        negative_index = np.random.choice(negative_indices)
        negative_num = self.img_labels.iloc[negative_index, 13]
        negative_num = str(negative_num)
        negative_img_path = os.path.join(self.img_dir, negative_num[:3], negative_num) + ".png"
        negative_image = Image.open(negative_img_path)

        if self.transform: # If a self transform function is specified, apply it
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image