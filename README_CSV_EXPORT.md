# CSV Export Guide

## Export DataFrame with All Columns to CSV

Use the `export_dataframe_csv.py` script to export data from ClickHouse to CSV with **ALL columns**.

### Usage

```bash
# Export first 10 rows with all columns (default)
python3 export_dataframe_csv.py ig_post

# Export specific number of rows
python3 export_dataframe_csv.py ig_post 100

# Export with custom filename
python3 export_dataframe_csv.py ig_post 10 my_data.csv

# Export from different tables
python3 export_dataframe_csv.py ig_comment 50
python3 export_dataframe_csv.py ig_users 20
```

### Examples

#### Basic Export (10 rows)
```bash
python3 export_dataframe_csv.py ig_post
```
Creates: `ig_post_sample_10rows_20251129_132557.csv`

#### Export More Rows
```bash
python3 export_dataframe_csv.py ig_post 100
```
Creates: `ig_post_sample_100rows_YYYYMMDD_HHMMSS.csv`

#### Custom Filename
```bash
python3 export_dataframe_csv.py ig_post 50 my_posts.csv
```

### Output

The CSV file includes:
- ✅ **ALL 65 columns** from the table
- ✅ Header row with column names
- ✅ UTF-8 encoding
- ✅ All data types preserved (as strings in CSV)

### Loading the CSV

#### In Python (pandas)
```python
import pandas as pd

# Load the CSV
df = pd.read_csv('ig_post_sample_10rows_20251129_132557.csv')

# Check shape
print(f"Shape: {df.shape}")  # (10, 65)

# View all columns
print(df.columns.tolist())

# View data
print(df.head())
```

#### In Python (csv module)
```python
import csv

with open('ig_post_sample_10rows_20251129_132557.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row['post_id'], row['like_count'])
```

#### In Excel/LibreOffice
- Just open the CSV file directly
- All 65 columns will be visible

### All 65 Columns Included

1. id
2. post_id
3. shortcode
4. typename
5. owner_ig_username
6. owner_ig_user_id
7. caption
8. caption_hashtags
9. caption_mentions
10. caption_urls
11. accessibility_caption
12. display_url
13. thumbnail_src
14. is_video
15. video_url
16. video_view_count
17. dimensions_height
18. dimensions_width
19. taken_at_timestamp
20. publication_timestamp
21. like_count
22. comment_count
23. location_id
24. location_name
25. is_paid_partnership
26. is_affiliate
27. comments_disabled
28. has_upcoming_event
29. viewer_has_liked
30. viewer_has_saved
31. viewer_can_reshare
32. is_carousel
33. carousel_media_count
34. tagged_users_count
35. tracking_token
36. source_type
37. raw_json
38. permalink
39. post_type
40. geotag_name
41. geotag_id
42. geotag_coordinates
43. audio_id
44. audio_title
45. audio_original_creator
46. audio_url
47. effect_name
48. effect_id
49. likes_visible
50. comments_visible
51. shares_visible
52. saves_visible
53. likers_list
54. commentators_list
55. is_pinned
56. media_dimensions
57. media_file_size
58. media_mime_type
59. media_exif
60. video_download_url
61. video_duration_s
62. video_metadata
63. polls_quizzes_questions
64. created_at
65. scraped_at

### Notes

- The CSV includes **all columns** - no data is filtered
- Large columns like `raw_json` are included as full text
- Arrays are exported as string representations
- Dates are in ISO format
- File encoding is UTF-8

