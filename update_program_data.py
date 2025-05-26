import pandas as pd
import requests
import humanize

API_URL = 'https://api.udacity.com/api/unified-catalog'
DATA_PATH = 'sample_programs.csv'


def convert_program_type(semantic_type: str) -> str:
    if semantic_type == 'Course':
        return 'FreeCourse'
    elif semantic_type == 'Degree':
        return 'Nanodegree'
    return 'PaidCourse'


def convert_slug_to_url(slug: str) -> str:
    return f"https://www.udacity.com/course/{slug}"


def convert_duration(duration_mins: float) -> str:
    if duration_mins and duration_mins > 0:
        delta = pd.Timedelta(minutes=duration_mins)
        return humanize.naturaldelta(delta)
    return 'Unknown duration'


def fetch_programs() -> pd.DataFrame:
    search_payload = {
        'PageSize': 1000,
        'SortBy': 'avgRating'
    }
    response = requests.post(f'{API_URL}/search', json=search_payload, timeout=10)
    response.raise_for_status()
    data = response.json()

    programs = []
    for item in data.get('searchResult', {}).get('hits', []):
        if not item.get('is_offered_to_public', False):
            continue
        duration_mins = float(item.get('duration', 0) or 0)
        programs.append({
            'key': item.get('key', ''),
            'program_type': convert_program_type(item.get('semantic_type', '')),
            'catalog_url': convert_slug_to_url(item.get('slug', '')),
            'duration': convert_duration(duration_mins),
            'difficulty': item.get('difficulty', 'Unknown'),
            'title': item.get('title', 'Untitled'),
            'summary': item.get('summary', ''),
            'skills': item.get('skill_names', [])
        })
    return pd.DataFrame(programs)


def main() -> None:
    df = fetch_programs()
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df)} programs to {DATA_PATH}")


if __name__ == '__main__':
    main()
