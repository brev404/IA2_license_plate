import re

def validate_plate(text):
  pattern = r'^[A-Z]{1,2}[0-9]{2,3}[A-Z]{1,2}$'  # Adjust for country formats
  return re.match(pattern, text) is not None
