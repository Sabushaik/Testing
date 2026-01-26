# Astec Video PDF Generator

This repository contains a script to generate a professional PDF document with Astec manufacturing video information.

## Generated Files

- **Astec Annotated.pdf** - The generated PDF containing a table with video names and their annotated presigned URLs

## Script

- **generate_astec_pdf.py** - Python script that creates the PDF

## Features

The generated PDF includes:
- Professional title: "Astec videos on the Manufacturing Pipeline with Nova pro model"
- Well-formatted table with:
  - Blue header with white text
  - Alternating row colors (white and light gray) for better readability
  - Proper borders and grid lines
  - Text alignment and padding
  - Clickable hyperlinks for each video URL
- All 10 Astec manufacturing videos with their presigned S3 URLs

## Requirements

```bash
pip install reportlab
```

## Usage

To regenerate the PDF:

```bash
python3 generate_astec_pdf.py
```

This will create/update the "Astec Annotated.pdf" file in the current directory.

## Video List

The PDF includes the following videos:
1. Bay6_fisheye.mp4
2. Bay6.mp4
3. Break_Press_Video.mp4
4. Break_Room_Exit.mp4
5. GEN_VIDEO_2(3).mp4
6. GEN_VIDEO_2(2).mp4
7. GEN_VIDEO_2.mp4
8. GEN_VIDEO_1.mp4
9. HP_Overview_2.mp4
10. HP_Overview_1.mp4

Each video entry includes a hyperlink to its annotated version hosted on AWS S3.
