import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(df, title='Class Distribution'):
    """
    Vẽ biểu đồ phân bố lớp.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x='category_str', data=df)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

def visualize_sample_images(df, num_samples=5):
    """
    Hiển thị một vài ảnh mẫu từ DataFrame.
    """
    import cv2
    import random
    
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        idx = random.randint(0, len(df)-1)
        row = df.iloc[idx]
        img_path = row['path_full']
        label = row['category_str']
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')
    plt.show()


def visualize_augmented_images(df, augment_transform, num_samples=5):
    """
    Hiển thị ảnh gốc và ảnh sau khi augmentation.
    
    Args:
        df: DataFrame chứa thông tin ảnh
        augment_transform: albumentations transform pipeline
        num_samples: số lượng ảnh mẫu
    """
    import cv2
    import random
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        idx = random.randint(0, len(df)-1)
        row = df.iloc[idx]
        img_path = row['path_full']
        label = row['category_str']
        
        # Đọc ảnh gốc
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Áp dụng augmentation
        augmented = augment_transform(image=img)
        img_aug = augmented['image']
        
        # Hiển thị ảnh gốc
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Original\n{label}')
        axes[0, i].axis('off')
        
        # Hiển thị ảnh augmented
        axes[1, i].imshow(img_aug)
        axes[1, i].set_title('Augmented')
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('Augmented', fontsize=12)
    
    plt.suptitle('Original vs Augmented Images', fontsize=14)
    plt.tight_layout()
    plt.show()
