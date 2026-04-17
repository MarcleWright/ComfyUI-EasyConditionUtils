# Easy Load Text Batch

## Attribution

This node is openly based on the batch-loading structure used by
`WASasquatchm` in `WAS Node Suite`, adapted here from image batch loading to
text batch loading for prompt testing workflows.

## Purpose

`Easy Load Text Batch` loads one text file from a directory and outputs its raw
content as a prompt string. It is intended for batch testing workflows, such as
evaluating the same LoRA or model against many prompt text files.

This node is the text-oriented counterpart to image batch loaders:

- it scans a directory by `path + pattern`
- it selects one file by mode
- it returns both the file content and the current filename

The implementation direction follows the general structure of WAS's
`Load Image Batch`, while changing the payload from image files to raw text
files.

## Inputs

- `mode`
  - `single_text`: use `index`
  - `incremental_text`: advance through the matched file list and loop at the end
  - `random`: select one file by `seed`
- `seed`
  - used only in `random`
- `index`
  - 0-based file index for `single_text`
- `label`
  - state key for `incremental_text`
- `path`
  - directory containing the text files
- `pattern`
  - glob pattern, default `*.txt`
- `encoding`
  - default `auto`
  - `auto` tries `utf-8-sig`, then `utf-8`, then `gb18030`
- `filename_text_extension`
  - `true`: output `case_01.txt`
  - `false`: output `case_01`

## Outputs

- `text`
  - the file content, unchanged
- `filename_text`
  - the current filename

## Behavior Notes

- text content is returned exactly as read from the file
- empty files are allowed and return an empty string
- no trimming or paragraph normalization is applied
- `incremental_text` loops back to the first file after the last file
- files are sorted by filename before selection

## Typical Use

- connect `text` into a prompt input
- connect `filename_text` into save-name or logging logic
- use `label` to keep independent incremental cursors for different test sets
