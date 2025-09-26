import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, color, filters, measure, morphology, transform
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Read and process the image
image = io.imread('car2.jpg')
# convert to grayscale
gray = color.rgb2gray(image)

# convert to black and white version 
thresh = filters.threshold_otsu(gray)
binary = gray < thresh

# Label connected regions (blobs of black pixels - invert binary for labeling)
# Since license plates appear as black regions in binary, we need to invert
binary_inverted = ~binary  # Invert so black becomes white for labeling
label_img = measure.label(binary_inverted)
regions = measure.regionprops(label_img)

# Find license plate candidates
candidates = []
for region in regions:
    minr, minc, maxr, maxc = region.bbox
    width, height = maxc - minc, maxr - minr
    aspect_ratio = width / height
    
    # Filter by aspect ratio and size
    if 2 < aspect_ratio < 6 and 1000 < region.area < 20000:
        candidates.append((minr, minc, maxr, maxc))

# Create a figure with subplots to display all images
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Display original image with rectangles around candidates
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image with License Plate Candidates")
axes[0, 0].axis('off')

# Draw rectangles around all candidates
for i, (minr, minc, maxr, maxc) in enumerate(candidates):
    # Create rectangle patch
    rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr, 
                           linewidth=2, edgecolor='red', facecolor='none')
    axes[0, 0].add_patch(rect)
    # Add candidate number
    axes[0, 0].text(minc, minr-5, f'Candidate {i+1}', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                   fontsize=10, color='red')

# Display grayscale image
axes[0, 1].imshow(gray, cmap='gray')
axes[0, 1].set_title("Grayscale Image")
axes[0, 1].axis('off')

# Display binary image
axes[1, 0].imshow(binary, cmap='gray')
axes[1, 0].set_title("Binary Image")
axes[1, 0].axis('off')

# Display the best candidate (first one) if any candidates were found
if candidates:
    minr, minc, maxr, maxc = candidates[0]
    plate_region = image[minr:maxr, minc:maxc]
    axes[1, 1].imshow(plate_region)
    axes[1, 1].set_title("Best License Plate Candidate")
    axes[1, 1].axis('off')
    
    # Create an unblurred version using image sharpening
    from skimage import restoration, exposure
    from scipy import ndimage
    
    # Convert plate to grayscale for processing
    plate_gray = color.rgb2gray(plate_region)
    
    # Apply unsharp mask for sharpening
    gaussian_blur = ndimage.gaussian_filter(plate_gray, sigma=1.0)
    unsharp_mask = plate_gray + 1.5 * (plate_gray - gaussian_blur)
    
    # Apply contrast enhancement
    unsharp_mask = exposure.rescale_intensity(unsharp_mask, out_range=(0, 1))
    
    # Convert back to RGB for saving
    if len(plate_region.shape) == 3:  # If original is color
        # Apply the enhancement to each channel
        plate_enhanced = np.zeros_like(plate_region, dtype=np.float64)
        for i in range(3):
            channel = plate_region[:, :, i] / 255.0
            gaussian_blur_ch = ndimage.gaussian_filter(channel, sigma=1.0)
            enhanced_ch = channel + 1.5 * (channel - gaussian_blur_ch)
            plate_enhanced[:, :, i] = exposure.rescale_intensity(enhanced_ch, out_range=(0, 1))
        
        # Convert to uint8 for saving
        plate_enhanced = (plate_enhanced * 255).astype(np.uint8)
    else:
        # Convert grayscale back to RGB
        plate_enhanced = (np.stack([unsharp_mask]*3, axis=-1) * 255).astype(np.uint8)
    
    # Save only the enhanced unblurred plate image
    io.imsave('plateImage.jpg', plate_enhanced)
    print(f"Enhanced unblurred plate image saved as 'plateImage.jpg'")
    print(f"Found {len(candidates)} license plate candidates")
else:
    axes[1, 1].text(0.5, 0.5, 'No candidates found', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=axes[1, 1].transAxes, fontsize=16)
    axes[1, 1].set_title("No License Plate Candidates Found")
    axes[1, 1].axis('off')
    print("No license plate candidates found")

plt.tight_layout()
plt.show()

