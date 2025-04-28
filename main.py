import os
import argparse
from utils import ensure_directory, print_step_info
from audio_processor import extract_audio_with_ffmpeg, transcribe_audio
from frame_processor import extract_frames, ocr_frames
from summarizer import summarize_text, smart_clean_summary_chunked
from quiz_generator import smart_real_mcq_generator

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Lecture Processing Pipeline")
    
    # Get the script directory to use as the base for default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument(
        "--video_path", 
        type=str, 
        default=os.path.join(script_dir, "video.mp4"),
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=os.path.join(script_dir, "output"),
        help="Directory for output files"
    )
    
    parser.add_argument(
        "--whisper_model", 
        type=str, 
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for transcription"
    )
    
    parser.add_argument(
        "--skip_frames", 
        action="store_true",
        help="Skip frame extraction and OCR"
    )
    
    parser.add_argument(
        "--fps", 
        type=int, 
        default=1,
        help="Frames per second to extract"
    )
    
    return parser.parse_args()


def main():
    """
    Main function to run the lecture processing pipeline
    """
    args = parse_arguments()
    
    # Ensure all paths are absolute
    video_path = os.path.abspath(args.video_path)
    output_dir = os.path.abspath(args.output_dir)
    
    # Create output directories
    output_dir = ensure_directory(output_dir)
    frames_dir = ensure_directory(os.path.join(output_dir, "frames"))
    
    # Define output file paths
    audio_path = os.path.join(output_dir, "audio.wav")
    transcription_path = os.path.join(output_dir, "transcription.txt")
    frames_list_path = os.path.join(output_dir, "frames.txt")
    ocr_results_path = os.path.join(output_dir, "ocr_results.txt")
    summary_path = os.path.join(output_dir, "summary.txt")
    cleaned_path = os.path.join(output_dir, "cleaned.txt")
    quiz_path = os.path.join(output_dir, "quiz.txt")
    
    # Step 1: Extract audio from video
    print_step_info("Audio Extraction")
    extract_audio_with_ffmpeg(video_path, audio_path)
    print_step_info("Audio Extraction", start=False)
    
    # Step 2: Transcribe audio
    print_step_info("Audio Transcription")
    transcription = transcribe_audio(audio_path, args.whisper_model, transcription_path)
    print_step_info("Audio Transcription", start=False)
    
    if not args.skip_frames:
        # Step 3: Extract frames for OCR
        print_step_info("Frame Extraction")
        frames = extract_frames(video_path, args.fps, frames_dir, frames_list_path)
        print_step_info("Frame Extraction", start=False)
        
        # Step 4: OCR on extracted frames
        print_step_info("OCR Processing")
        ocr_results = ocr_frames(frames_dir, ocr_results_path)
        print_step_info("OCR Processing", start=False)
    
    # Step 5: Summarize the transcription
    print_step_info("Text Summarization")
    summary = summarize_text(transcription, summary_path)
    print_step_info("Text Summarization", start=False)
    
    # Step 6: Clean the summary
    print_step_info("Summary Cleaning")
    smart_clean_summary_chunked(summary_path, cleaned_path)
    print_step_info("Summary Cleaning", start=False)
    
    # Step 7: Generate MCQs
    print_step_info("Quiz Generation")
    smart_real_mcq_generator(cleaned_path, quiz_path)
    print_step_info("Quiz Generation", start=False)
    
    print("\nLecture processing completed! Output files:")
    print(f"- Transcription: {transcription_path}")
    print(f"- Summary: {summary_path}")
    print(f"- Cleaned Summary: {cleaned_path}")
    print(f"- Quiz: {quiz_path}")


if __name__ == "__main__":
    main()