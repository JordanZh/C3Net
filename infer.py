from core.utils import audio_image2text_inference, audio_text2image_inference, image_text2audio_inference, get_inference_model
import argparse


if __name__ == "__main__":
    # From argpaeser
    # Usage: python inference.py --audio_path <path> --image_path <path> --text <text> --save_path <path> --mode <mode>
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, default=None, help='Path to the audio file')
    parser.add_argument('--image_path', type=str, default=None, help='Path to the image file')
    parser.add_argument('--text', type=str, default=None, help='Text to be used for inference')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the output')
    parser.add_argument('--mode', type=str, choices=['audio_image2text', 'audio_text2image', 'image_text2audio'], required=True, help='Inference mode')
    args = parser.parse_args()
    inference_tester = get_inference_model()
    if args.mode == 'audio_image2text':
        args.save_path = args.save_path if args.save_path else 'result/audio_image2text.txt'
        args.audio_path = args.audio_path if args.audio_path else 'examples/guitar_sound.flac'
        args.image_path = args.image_path if args.image_path else 'examples/rain_princess_style.jpg'
        audio_image2text_inference(
            inference_tester,
            save_path=args.save_path,
            audio_path=args.audio_path,
            image_path=args.image_path
        )
    elif args.mode == 'audio_text2image':
        args.save_path = args.save_path if args.save_path else 'result/audio_text2image.jpg'
        args.audio_path = args.audio_path if args.audio_path else 'examples/sea_waves.wav'
        args.text = args.text if args.text else 'under golden sun set'
        audio_text2image_inference(
            inference_tester,
            save_path=args.save_path,
            audio_path=args.audio_path,
            text=args.text
        )
    elif args.mode == 'image_text2audio':
        args.save_path = args.save_path if args.save_path else 'result/image_text2audio.wav'
        args.image_path = args.image_path if args.image_path else 'examples/room.jpg'
        args.text = args.text if args.text else 'people are talking'
        image_text2audio_inference(
            inference_tester,
            save_path=args.save_path,
            image_path=args.image_path,
            text=args.text
        )