#!/usr/bin/env python3
import argparse
import logging
from datetime import datetime
from pathlib import Path
from . import create_anemia_explainer, ExplanationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Anemia Detection LIME Explainer')
    parser.add_argument('--weights', required=True, help='Path to model weights')
    parser.add_argument('--image', required=True, help='Image path or directory')
    parser.add_argument('--output', help='Output directory for explanations')
    parser.add_argument('--device', help='Device to use (cuda/cpu)')
    parser.add_argument('--samples', type=int, default=1000, help='LIME samples')
    parser.add_argument('--features', type=int, default=8, help='Features to show')
    parser.add_argument('--batch', action='store_true', help='Batch processing mode')

    args = parser.parse_args()

    try:
        start_time = datetime.now()

        # Create output directory
        output_dir = None
        if args.output:
            output_dir = Path(args.output) / f"explanations_{start_time:%Y%m%d_%H%M%S}"
            output_dir.mkdir(parents=True, exist_ok=True)

        # Create explainer
        explainer = create_anemia_explainer(
            weights_path=args.weights,
            device=args.device,
            num_samples=args.samples,
            num_features=args.features
        )

        if args.batch:
            # Process directory
            from glob import glob
            image_paths = glob(str(Path(args.image) / '*.*'))
            if not image_paths:
                raise ValueError(f"No images found in {args.image}")

            results = explainer.explain_multiple_images(
                image_paths=image_paths,
                save_dir=str(output_dir) if output_dir else None
            )

            success = sum(1 for r in results if 'error' not in r)
            print(f"Processed {success}/{len(image_paths)} images successfully")
        else:
            # Process single image
            result = explainer.explain_single_image(
                image_path=args.image,
                save_path=str(output_dir / "explanation.png") if output_dir else None
            )

            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Explanation generated. Predicted: {result['class_name']}")

        # Performance stats
        elapsed = datetime.now() - start_time
        print(f"Total time: {elapsed.total_seconds():.2f} seconds")

    except ExplanationError as ee:
        logger.error(f"Explanation error: {ee}")
        print(f"ERROR: {ee}")
        return 1
    except Exception as e:
        logger.exception("Critical error")
        print(f"CRITICAL ERROR: {e}")
        return 2
    finally:
        if 'explainer' in locals():
            explainer.cleanup()

        if args.device and 'cuda' in args.device:
            import torch
            torch.cuda.empty_cache()

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())