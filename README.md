# Astec Transcripts PDF Generator

This repository contains a script to generate a beautifully formatted PDF from the Astec Transcripts.txt file.

## Features

- Converts video transcripts to a professional PDF format
- Removes `<>` symbols and formats the content inside as styled headings
- Maintains proper spacing between different video sections
- Includes:
  - Main title centered and prominent
  - Video names as bold section headers
  - Tag names (general, machine_type, worker_safety, operations, anomalies) formatted as subheadings
  - Time ranges clearly marked
  - Body text justified and properly indented

## Requirements

```bash
pip install reportlab
```

## Usage

### Default Usage
Simply run the script in the same directory as `Astec Transcripts.txt`:

```bash
python generate_pdf.py
```

This will generate `Astec Transcripts.pdf` in the same directory.

### Custom Input/Output Files

```bash
python generate_pdf.py input_file.txt output_file.pdf
```

## Output

The generated PDF (`Astec Transcripts.pdf`) is a professionally formatted 19-page document containing all the video transcripts with proper alignment and styling.

## Files

- `Astec Transcripts.txt` - Original transcript file
- `generate_pdf.py` - PDF generation script
- `Astec Transcripts.pdf` - Generated PDF output
- `code.py` - FastAPI application (existing)
