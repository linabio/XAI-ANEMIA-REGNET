import gradio as gr
from explainability.lime_explainer import create_anemia_explainer
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import tempfile
import io
from PIL import Image
import socket
import argparse
import cv2
from segmentation.unet import segment_image_with_unet, apply_segmentation_mask

MODEL_PATH = os.getenv("ANEMIA_MODEL_PATH", "classification/best_regnet_anemia_classifier.pth")
UNET_MODEL_PATH = os.getenv("UNET_MODEL_PATH", "segmentation/best_unet_model.pth")

def analyze_anemia(image, num_samples, num_features, use_gpu, use_unet_segmentation, 
                  post_process_unet, min_area_ratio, kernel_size, remove_small_components, expand_borders):
    yield None, "", "⏳ Procesando..."

    try:
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        status = f"✅ Usando {'GPU' if device == 'cuda' else 'CPU'}"

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            image.save(temp_file.name)
            temp_image_path = temp_file.name

        segmented_image_path = temp_image_path
        mask = None  
        
        if use_unet_segmentation:
            yield None, "", "🔍 Segmentando imagen con U-Net..."
            
            if not os.path.exists(UNET_MODEL_PATH):
                yield None, "❌ Modelo U-Net no encontrado", f"Error: Modelo U-Net no encontrado en {UNET_MODEL_PATH}"
                return
            
            result = segment_image_with_unet(
                temp_image_path, 
                UNET_MODEL_PATH, 
                device,
                post_process=post_process_unet,
                min_area_ratio=min_area_ratio,
                kernel_size=kernel_size,
                remove_small_components=remove_small_components,
                expand_borders=expand_borders
            )
            if result is None:
                yield None, "❌ Error en segmentación U-Net", "Error: Fallo en segmentación"
                return
            
            segmented_image, mask = result
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as seg_temp_file:
                segmented_image_pil = Image.fromarray(segmented_image.astype(np.uint8))
                segmented_image_pil.save(seg_temp_file.name)
                segmented_image_path = seg_temp_file.name
            
            status = f"✅ Segmentación U-Net completada. Usando {'GPU' if device == 'cuda' else 'CPU'}"

        yield None, "", "🔄 Ejecutando análisis LIME..."
        
        explainer = create_anemia_explainer(
            weights_path=MODEL_PATH,
            device=device,
            num_samples=num_samples,
            num_features=num_features
        )

        original_img = np.array(Image.open(temp_image_path))
        
        from segmentation.unet import apply_lime_to_cropped_region
        
        if use_unet_segmentation and mask is not None:
            result = apply_lime_to_cropped_region(
                segmented_image=segmented_image,
                mask=mask,
                explainer=explainer
            )
        else:
            result = explainer.explain_single_image(
                image_path=temp_image_path,
                save_path=None,
                show_plot=False
            )
            
            if result and 'explanation' in result and result['explanation'] is not None:
                explanation = result['explanation']
                try:
                    _, mask_pos = explanation.get_image_and_mask(
                        explanation.top_labels[0],
                        positive_only=True,
                        num_features=num_features,
                        hide_rest=False
                    )
                    
                    _, mask_all = explanation.get_image_and_mask(
                        explanation.top_labels[0],
                        positive_only=False,
                        num_features=num_features,
                        hide_rest=False
                    )
                    
                    mask_neg = (mask_all & ~mask_pos).astype(bool)
                    
                    result['mask_positive'] = mask_pos
                    result['mask_negative'] = mask_neg
                except Exception as e:
                    print(f"⚠️ Error generando máscaras: {e}")
                    result['mask_positive'] = np.zeros(original_img.shape[:2], dtype=bool)
                    result['mask_negative'] = np.zeros(original_img.shape[:2], dtype=bool)
            else:
                result = {
                    'image': original_img,
                    'mask_positive': np.zeros(original_img.shape[:2], dtype=bool),
                    'mask_negative': np.zeros(original_img.shape[:2], dtype=bool),
                    'class_name': 'Error en análisis',
                    'error': 'No se pudo generar explicación'
                }

        lime_image = result['image']  
        mask_positive = result.get('mask_positive', np.zeros(result['image'].shape[:2], dtype=bool))
        mask_negative = result.get('mask_negative', np.zeros(result['image'].shape[:2], dtype=bool))
        
        from skimage.segmentation import mark_boundaries
        import matplotlib.patches as mpatches
        
        original_img = np.array(Image.open(temp_image_path))
        original_size = (original_img.shape[1], original_img.shape[0]) 
        
        if lime_image.shape[:2] != original_img.shape[:2]:
            print(f"🔧 Redimensionando imagen LIME de {lime_image.shape[:2]} a {original_img.shape[:2]}")
            lime_image = cv2.resize(lime_image, original_size)
            mask_positive = cv2.resize(mask_positive.astype(np.float32), original_size) > 0.5
            mask_negative = cv2.resize(mask_negative.astype(np.float32), original_size) > 0.5
        
        if use_unet_segmentation:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
            
            ax1.imshow(original_img)
            ax1.set_title('📷 Imagen Original', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            if use_unet_segmentation and segmented_image_path != temp_image_path:
                segmented_img = np.array(Image.open(segmented_image_path))
                ax2.imshow(segmented_img)
                ax2.set_title('🎯 Imagen Segmentada (U-Net)', fontsize=14, fontweight='bold')
                ax2.axis('off')
            else:
                ax2.imshow(original_img)
                ax2.set_title('🎯 Sin Segmentación', fontsize=14, fontweight='bold')
                ax2.axis('off')
            
            if 'bbox' in result:
                bbox = result['bbox']
                cropped_region = original_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                ax3.imshow(cropped_region)
                ax3.set_title(f'✂️ Región Recortada\n{bbox[2]-bbox[0]}x{bbox[3]-bbox[1]} píxeles', fontsize=14, fontweight='bold')
                ax3.axis('off')
            else:
                ax3.imshow(original_img)
                ax3.set_title('✂️ Sin Recorte', fontsize=14, fontweight='bold')
                ax3.axis('off')
            
            ax4.imshow(lime_image)
            
            if mask_positive is not None and mask_positive.any():
                positive_overlay = np.zeros_like(lime_image)
                positive_overlay[mask_positive] = [0, 1, 0]  
                ax4.imshow(positive_overlay, alpha=0.3)
            
            if mask_negative is not None and mask_negative.any():
                negative_overlay = np.zeros_like(lime_image)
                negative_overlay[mask_negative] = [1, 0, 0]  
                ax4.imshow(negative_overlay, alpha=0.3)
            
            ax4.set_title('🔍 Explicación LIME', fontsize=14, fontweight='bold')
            ax4.axis('off')
            
            legend_elements = []
            if mask_positive is not None and mask_positive.any():
                legend_elements.append(mpatches.Patch(color='green', alpha=0.7, label='✅ Refuerzo Positivo'))
            if mask_negative is not None and mask_negative.any():
                legend_elements.append(mpatches.Patch(color='red', alpha=0.7, label='❌ Refuerzo Negativo'))
            
            if legend_elements:
                ax4.legend(handles=legend_elements, loc='upper right', fontsize=10)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            ax1.imshow(lime_image)
            
            if mask_positive is not None and mask_positive.any():
                positive_overlay = np.zeros_like(lime_image)
                positive_overlay[mask_positive] = [0, 1, 0]  # Verde
                ax1.imshow(positive_overlay, alpha=0.3)
            
            if mask_negative is not None and mask_negative.any():
                negative_overlay = np.zeros_like(lime_image)
                negative_overlay[mask_negative] = [1, 0, 0]  # Rojo
                ax1.imshow(negative_overlay, alpha=0.3)
            
            ax1.set_title('🔍 Explicación LIME con Colores', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            legend_elements = []
            if mask_positive is not None and mask_positive.any():
                legend_elements.append(mpatches.Patch(color='green', alpha=0.7, label='✅ Refuerzo Positivo'))
            if mask_negative is not None and mask_negative.any():
                legend_elements.append(mpatches.Patch(color='red', alpha=0.7, label='❌ Refuerzo Negativo'))
            
            if legend_elements:
                ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            combined_mask = np.zeros(lime_image.shape[:2], dtype=bool)
            if mask_positive is not None:
                combined_mask |= mask_positive
            if mask_negative is not None:
                combined_mask |= mask_negative
            
            lime_image_with_boundaries = mark_boundaries(lime_image, combined_mask)
            ax2.imshow(lime_image_with_boundaries)
            ax2.set_title('📐 Límites de Segmentación', fontsize=14, fontweight='bold')
            ax2.axis('off')
        
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)

        img.save("debug_gradio_output.png")

        os.remove(temp_image_path)
        if use_unet_segmentation and segmented_image_path != temp_image_path:
            os.remove(segmented_image_path)

        yield img, result['class_name'], status

    except FileNotFoundError as e:
        yield None, f"❌ Modelo no encontrado: {e}", "Error: Modelo no encontrado"
    except ValueError as e:
        yield None, f"❌ Imagen inválida: {e}", "Error: Imagen inválida"
    except Exception as e:
        yield None, f"❌ Error inesperado: {str(e)}", f"Error: {str(e)}"

def show_feedback_form():
    return gr.update(visible=True)

def submit_feedback(feedback):
    return "✅ ¡Gracias por tu comentario!"

with gr.Blocks(title="Anemia Analyzer") as app:
    gr.Markdown("# 🧨 Analizador de Anemia con LIME")
    gr.Markdown("Sube una imagen de la conjuntiva palpebral para detectar posibles signos de anemia. Este sistema usa IA y explicaciones visuales con LIME.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📄 Subir Imagen")
            image_input = gr.Image(
                label="Imagen de conjuntiva",
                type="pil",
                elem_id="image-upload",
                height=300
            )
            gr.Markdown("💡 Usa imágenes JPG o PNG claras y bien iluminadas.")

            with gr.Accordion("⚙️ Opciones Avanzadas", open=False):
                num_samples = gr.Slider(
                    100, 2000, value=1000, step=100,
                    label="Número de muestras para LIME"
                )
                gr.Markdown("🔬 Más muestras = más precisión, pero más tiempo.")

                num_features = gr.Slider(
                    1, 10, value=5, step=1,
                    label="Número de regiones importantes"
                )
                gr.Markdown("📌 Define cuántas regiones relevantes se resaltarán.")

                use_gpu = gr.Checkbox(value=True, label="Usar GPU si está disponible")
                
                use_unet_segmentation = gr.Checkbox(
                    value=True, 
                    label="Usar segmentación U-Net antes del análisis"
                )
                gr.Markdown("🎯 Segmenta la imagen con U-Net antes del análisis LIME para mejor precisión.")
                
                post_process_unet = gr.Checkbox(
                    value=True,
                    label="Post-procesar segmentación U-Net"
                )
                gr.Markdown("🔧 Elimina islas pequeñas y mejora la conectividad de la segmentación.")
                
                min_area_ratio = gr.Slider(
                    0.0001, 0.05, value=0.001, step=0.0001,
                    label="Área mínima de segmentación (ratio)"
                )
                gr.Markdown("📏 Define el tamaño mínimo de la región segmentada (0.1% = 0.001, más pequeño = menos agresivo).")
                
                kernel_size = gr.Slider(
                    3, 15, value=3, step=2,
                    label="Tamaño del kernel morfológico"
                )
                gr.Markdown("🔲 Tamaño del kernel para operaciones morfológicas (3 = suave, 15 = muy suavizado).")
                
                remove_small_components = gr.Checkbox(
                    value=True,
                    label="Eliminar componentes pequeños"
                )
                gr.Markdown("🎯 Mantener solo la región más grande (desactivar para mantener todas las regiones).")
                
                expand_borders = gr.Slider(
                    0, 20, value=5, step=1,
                    label="Expandir bordes (píxeles)"
                )
                gr.Markdown("🔲 Expandir los bordes de la segmentación (0 = sin expansión, 20 = máxima expansión).")

            submit_btn = gr.Button("🚀 Analizar Imagen", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("## 📊 Resultados")
            image_output = gr.Image(label="Explicación LIME", interactive=False, height=300)
            diagnosis_output = gr.Textbox(label="🧪 Diagnóstico", interactive=False)
            status_output = gr.Textbox(label="📱 Estado del análisis", value="Listo para analizar", interactive=False)

    example_dir = Path("ui/examples") if Path("ui/examples").exists() else Path("sample_images")
    if example_dir.exists():
        example_files = list(example_dir.glob("*.jpg"))
        if len(example_files) >= 2:
            gr.Examples(
                examples=[
                    [str(example_dir / "normal.jpg"), 1000, 5, True, True, True, 0.001, 3, True, 5],
                    [str(example_dir / "anemic.jpg"), 800, 6, False, True, True, 0.001, 3, True, 5]
                ],
                inputs=[image_input, num_samples, num_features, use_gpu, use_unet_segmentation, 
                       post_process_unet, min_area_ratio, kernel_size, remove_small_components, expand_borders],
                outputs=[image_output, diagnosis_output, status_output],
                label="📂 Ejemplos de prueba"
            )

    gr.Markdown("---")
    with gr.Row():
        gr.Markdown("¿Tienes sugerencias o comentarios?")
        feedback_btn = gr.Button("📝 Enviar Feedback")

    feedback_output = gr.Textbox(
        label="✍️ Escribe tu comentario",
        placeholder="Tu sugerencia será bienvenida",
        interactive=True,
        visible=False
    )

    feedback_btn.click(show_feedback_form, outputs=feedback_output)
    feedback_output.change(submit_feedback, inputs=feedback_output, outputs=feedback_output)

    submit_btn.click(
        fn=analyze_anemia,
        inputs=[image_input, num_samples, num_features, use_gpu, use_unet_segmentation, 
               post_process_unet, min_area_ratio, kernel_size, remove_small_components, expand_borders],
        outputs=[image_output, diagnosis_output, status_output]
    )

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        return s.getsockname()[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anemia Analyzer")
    parser.add_argument("--port", type=int, default=None, help="Puerto para la aplicación")
    args = parser.parse_args()
    port = args.port or find_free_port()

    print(f"🌐 Aplicación ejecutándose en: http://localhost:{port}")
    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=True,
        show_error=True,
        inbrowser=True,
        auth=("1", "1")
    )
