# Lecture Processing Pipeline

A comprehensive tool for processing lecture videos to generate transcripts, summaries, and quizzes.

## Overview

This pipeline processes a lecture video and outputs:

- A full transcript of the lecture
- A summary of important academic content
- A cleaned version focusing purely on educational material
- An auto-generated quiz with realistic MCQs based on the content

The pipeline runs entirely offline, uses GPU acceleration where possible, and supports caching to avoid redundant computation.

## Features

- End-to-end offline processing: no API calls or external server dependencies
- GPU acceleration for transcription, summarization, OCR, and question generation
- Caching at every major step to avoid recomputation
- Redundancy filtering for frames to save processing time
- Realistic MCQs with well-formed questions and answer options

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/lecture-processor.git
cd lecture-processor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py --video_path path/to/lecture.mp4
```

### Advanced Options

```bash
python main.py --video_path path/to/lecture.mp4 --output_dir custom_output --whisper_model small --fps 2
```

### Command Line Arguments

- `--video_path`: Path to input video file (default: "video.mp4")
- `--output_dir`: Directory for output files (default: "output")
- `--whisper_model`: Whisper model size for transcription ("tiny", "base", "small", "medium", "large") (default: "tiny")
- `--skip_frames`: Skip frame extraction and OCR
- `--fps`: Frames per second to extract (default: 1)

## Output Files

- `audio.wav`: Extracted lecture audio
- `transcription.txt`: Full lecture transcript
- `frames.txt`: List of extracted frames for OCR
- `ocr_results.txt`: Text detected from lecture slides/board
- `summary.txt`: Summarized transcript
- `cleaned.txt`: Academic content only
- `quiz.txt`: MCQs based on lecture content

## Example

1. Process a lecture video:
```bash
python main.py --video_path lectures/machine_learning_intro.mp4 --whisper_model small
```

2. View the generated quiz:
```bash
cat output/quiz.txt
```
 
