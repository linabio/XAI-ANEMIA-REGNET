import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from scipy import ndimage
from skimage import morphology
import tempfile

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = nn.Sequential(CBR(3, 64), CBR(64, 64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(CBR(128, 256), CBR(256, 256))

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 64))
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return torch.sigmoid(self.final(d1))


def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice


def preprocess_image_for_unet(image_path, target_size=(256, 256), normalize_background=True):
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = np.array(image_path)
        if len(image.shape) == 3 and image.shape[2] == 3:
            pass
        else:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if normalize_background:
        image = normalize_white_background(image)
    
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image_tensor


def normalize_white_background(image):

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("La imagen debe ser RGB con 3 canales")
    
    img_rgb = image.astype(np.uint8)
    
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 0, 180])  # blancos no puros
    upper_white = np.array([180, 50, 255])
    mask_hsv = cv2.inRange(hsv, lower_white, upper_white)

    lower_white_rgb = np.array([200, 200, 200])
    upper_white_rgb = np.array([255, 255, 255])
    mask_rgb = cv2.inRange(img_rgb, lower_white_rgb, upper_white_rgb)

    mask = cv2.bitwise_or(mask_hsv, mask_rgb)

    mask = cv2.bitwise_not(mask)

    img_roi = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    img_with_alpha = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2RGBA)
    img_with_alpha[:, :, 3] = mask
    
    background = np.zeros_like(img_rgb)
    alpha = img_with_alpha[:, :, 3:4] / 255.0
    normalized_image = img_with_alpha[:, :, :3] * alpha + background * (1 - alpha)
    
    return normalized_image.astype(np.uint8)


def create_segmented_image_with_black_background(image, mask):

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("La imagen debe ser RGB con 3 canales")
    
    if mask.shape[:2] != image.shape[:2]:
        raise ValueError("La máscara debe tener las mismas dimensiones que la imagen")
    
    image_uint8 = image.astype(np.uint8)
    
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    segmented_image = cv2.bitwise_and(image_uint8, image_uint8, mask=mask_uint8)
    
    return segmented_image


def create_white_background_image(image_with_black_bg):

    if len(image_with_black_bg.shape) != 3 or image_with_black_bg.shape[2] != 3:
        raise ValueError("La imagen debe ser RGB con 3 canales")
    
    white_background = np.ones_like(image_with_black_bg) * 255
    
    roi_mask = np.any(image_with_black_bg > 20, axis=2)
    
    roi_mask = roi_mask.astype(np.uint8) * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
    
    roi_mask = roi_mask > 0
    
    result = white_background.copy()
    result[roi_mask] = image_with_black_bg[roi_mask]
    
    return result.astype(np.uint8)


def expand_mask_borders(mask, expansion_pixels=5):

    mask = (mask * 255).astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expansion_pixels*2+1, expansion_pixels*2+1))
    
    expanded_mask = cv2.dilate(mask, kernel, iterations=1)
    
    return expanded_mask.astype(np.float32) / 255.0


def post_process_mask(mask, min_area_ratio=0.001, kernel_size=3, remove_small_components=True, expand_borders=0):

    mask = (mask * 255).astype(np.uint8)
    
    h, w = mask.shape
    mask_filled = np.zeros((h+2, w+2), dtype=np.uint8)
    mask_filled[1:h+1, 1:w+1] = mask
    
    cv2.floodFill(mask_filled, None, (0, 0), 255)
    
    holes = cv2.bitwise_not(mask_filled)
    
    holes = holes[1:h+1, 1:w+1]
    
    mask = cv2.bitwise_or(mask, holes)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size+2, kernel_size+2))
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)
    
    mask = cv2.erode(mask, kernel_dilate, iterations=1)
    
    if not remove_small_components:
        processed_mask = mask.astype(np.float32) / 255.0
    else:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels <= 1:
            processed_mask = mask.astype(np.float32) / 255.0
        else:            areas = stats[1:, cv2.CC_STAT_AREA]  # Excluir el primer componente (fondo)
            if len(areas) == 0:
                return np.zeros_like(mask, dtype=np.float32)
            
            largest_component = np.argmax(areas) + 1  # +1 porque excluimos el fondo
            
            total_area = mask.shape[0] * mask.shape[1]
            min_area = total_area * min_area_ratio
            
            if areas[largest_component - 1] < min_area:
                print(f"⚠️ Componente más grande ({areas[largest_component - 1]} píxeles) es menor que el mínimo ({min_area:.0f}). Manteniendo máscara original.")
                processed_mask = mask.astype(np.float32) / 255.0
            else:
                clean_mask = np.zeros_like(mask)
                clean_mask[labels == largest_component] = 255
                
                clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
                processed_mask = clean_mask.astype(np.float32) / 255.0
    
    if expand_borders > 0:
        processed_mask = expand_mask_borders(processed_mask, expand_borders)
        print(f"🔲 Bordes expandidos en {expand_borders} píxeles")
    
    return processed_mask


def segment_image_with_unet(image_path, model_path, device='cpu', target_size=(256, 256), 
                           post_process=True, min_area_ratio=0.001, kernel_size=3, 
                           remove_small_components=True, expand_borders=0, normalize_background=True):
    model = UNet().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ Modelo UNet cargado exitosamente")
    except Exception as e:
        print(f"❌ Error cargando modelo UNet: {e}")
        return None
    
    image_tensor = preprocess_image_for_unet(image_path, target_size, normalize_background)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        mask = model(image_tensor)
        binary_mask = (mask > 0.5).float()
    
    mask_np = binary_mask.cpu().squeeze().numpy()
    
    if post_process:
        print(f"🔧 Aplicando post-procesamiento (área_min: {min_area_ratio}, kernel: {kernel_size}, expand: {expand_borders})...")
        mask_np = post_process_mask(mask_np, min_area_ratio, kernel_size, remove_small_components, expand_borders)
        print(f"   Píxeles segmentados después del post-procesamiento: {np.sum(mask_np):.0f}")
    
    if isinstance(image_path, str):
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_size = (original_image.shape[1], original_image.shape[0])
    else:
        original_image = np.array(image_path)
        original_size = (original_image.shape[1], original_image.shape[0])
    
    if original_size != target_size:
        mask_np = cv2.resize(mask_np.astype(np.float32), original_size)
        mask_np = (mask_np > 0.5).astype(np.float32)
    
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
    mask_np = mask_closed.astype(np.float32) / 255.0
    
    segmented_image = create_segmented_image_with_black_background(original_image, mask_np)
    
    return segmented_image, mask_np


def apply_segmentation_mask(image, mask, darken_background=False):
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif hasattr(image, 'save'):
        image = np.array(image)
        if len(image.shape) == 3 and image.shape[2] == 3:
            pass  # Ya está en RGB
        else:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    segmented_image = image.copy()
    
    if darken_background:
        segmented_image[mask == 0] = 0
    else:

        background_alpha = 0.3
        background_alpha = 0.3
        segmented_image = segmented_image.astype(np.float32)
        
        background_mask = (mask == 0)
        segmented_image[background_mask] = segmented_image[background_mask] * background_alpha
        
        foreground_mask = (mask > 0)
        segmented_image[foreground_mask] = image[foreground_mask]
        
        segmented_image = segmented_image.astype(np.uint8)
    
    return segmented_image


def crop_segmented_image(image, mask, padding=10):

    if mask is not None and mask.shape[:2] == image.shape[:2]:
        coords = np.where(mask > 0.5)
    else:
        if len(image.shape) == 3:
         
            non_black_pixels = np.any(image > 10, axis=2)
        else:
            non_black_pixels = image > 10
        coords = np.where(non_black_pixels)
    
    if len(coords[0]) == 0:
        return image, (0, 0, image.shape[1], image.shape[0])
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    y_min = max(0, y_min - padding)
    y_max = min(image.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image.shape[1], x_max + padding)
    
    cropped_image = image[y_min:y_max, x_min:x_max]
    
    if cropped_image.shape[-1] == 4:
        cropped_image = cropped_image[:, :, :3]
    
    bbox = (x_min, y_min, x_max, y_max)
    
    return cropped_image, bbox


def apply_lime_to_cropped_region(segmented_image, mask, explainer, target_size=(224, 224), normalize_background=True):
  
    cropped_image, bbox = crop_segmented_image(segmented_image, mask)
    
    print(f"🔧 Imagen recortada: {cropped_image.shape}, bbox: {bbox}")
    
    white_background_image = create_white_background_image(cropped_image)
    
    if normalize_background:
        print("🔧 Aplicando normalización de fondo blanco a imagen recortada...")
        white_background_image = normalize_white_background(white_background_image)
        print(f"   Imagen normalizada: {white_background_image.shape}")
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        cropped_image_pil = Image.fromarray(white_background_image.astype(np.uint8))
        cropped_image_pil.save(temp_file.name)
        cropped_image_path = temp_file.name
    
    try:
        result = explainer.explain_single_image(
            image_path=cropped_image_path,
            save_path=None,
            show_plot=False
        )
        
        print(f"🧠 LIME result keys: {result.keys() if result else 'None'}")
        
        if result is None or 'error' in result:
            print(f"❌ Error en LIME: {result.get('error', 'Unknown error') if result else 'No result'}")
            return {
                'image': white_background_image,
                'mask_positive': np.zeros(white_background_image.shape[:2], dtype=bool),
                'mask_negative': np.zeros(white_background_image.shape[:2], dtype=bool),
                'class_name': 'Error en análisis LIME',
                'bbox': bbox,
                'error': result.get('error', 'LIME failed') if result else 'No LIME result'
            }
        
        if 'explanation' in result and result['explanation'] is not None:
            explanation = result['explanation']
            
            try:
                _, mask_pos = explanation.get_image_and_mask(
                    explanation.top_labels[0],
                    positive_only=True,
                    num_features=5,
                    hide_rest=False
                )
                
                _, mask_all = explanation.get_image_and_mask(
                    explanation.top_labels[0],
                    positive_only=False,
                    num_features=5,
                    hide_rest=False
                )
                
                mask_neg = (mask_all & ~mask_pos).astype(bool)
                
                result['mask_positive'] = mask_pos
                result['mask_negative'] = mask_neg
                
                print(f"✅ Máscaras generadas - Positivas: {np.sum(mask_pos)}, Negativas: {np.sum(mask_neg)}")
                
            except Exception as e:
                print(f"⚠️ Error generando máscaras desde explicación: {e}")
                result['mask_positive'] = np.zeros(cropped_image.shape[:2], dtype=bool)
                result['mask_negative'] = np.zeros(cropped_image.shape[:2], dtype=bool)
        else:
            print("⚠️ No se encontró explicación en el resultado")
            result['mask_positive'] = np.zeros(cropped_image.shape[:2], dtype=bool)
            result['mask_negative'] = np.zeros(cropped_image.shape[:2], dtype=bool)
        
        if 'image' not in result or result['image'] is None:
            print("⚠️ No se encontró imagen en resultado, usando imagen recortada")
            result['image'] = white_background_image
        
        result['bbox'] = bbox
        result['cropped_image_path'] = cropped_image_path
        
        return result
        
    except Exception as e:
        print(f"❌ Error aplicando LIME a región recortada: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'image': white_background_image,
            'mask_positive': np.zeros(white_background_image.shape[:2], dtype=bool),
            'mask_negative': np.zeros(white_background_image.shape[:2], dtype=bool),
            'class_name': f'Error en análisis: {str(e)}',
            'bbox': bbox,
            'error': str(e)
        } 
