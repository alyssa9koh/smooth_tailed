# meow hello

class TaxaDataset(Dataset):
      def __init__(self, annotations_file, positive_json, negative_json, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        with open(positive_json, 'r') as file:
          positive_data = file.read()
        self.positive_indices_dict = json.loads(positive_data)
        with open(negative_json, 'r') as file:
          negative_data = file.read()
        self.negative_indices_dict = json.loads(negative_data)
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
        anchor_label = self.img_labels.iloc[idx, 10] # Getting label; this is the species name. Taxon_id is index number 4

        # select positive sample with the same label as anchor
        # Be sure that the similarity score is 8 or higher.
        positive_indices = np.array(self.positive_indices_dict[anchor_label])
        positive_index = np.random.choice(positive_indices)
        positive_num = self.img_labels.iloc[positive_index, 13]
        positive_num = str(positive_num)
        positive_img_path = os.path.join(self.img_dir, positive_num[:3], positive_num) + ".png"
        positive_image = Image.open(positive_img_path)

        # getting our negative image.
        # Be sure that the similarity score is 7 or lower.
        negative_indices = np.array(self.negative_indices_dict[anchor_label])
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