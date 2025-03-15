def read_json(data_path, key=None):
    """Function to read json data"""
    import json

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []
    if key is not None:
        for i, entry in enumerate(data):
            if "FactualNarrative" in entry:
                raw = entry["FactualNarrative"]
                # Need to drop long reports, as they will cause errors in the LLM query.
                if raw is not None and len(raw) <= 25000:
                    processed_data.append(raw)
                else:
                    processed_data.append(None)

    else:
        return processed_data

    return processed_data
