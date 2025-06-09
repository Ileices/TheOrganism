import json

def validate_form(form_data):
    errors = []
    if not form_data.get('age'):
        errors.append('Age is required.')
    elif not isinstance(form_data.get('age'), int):
        errors.append('Age must be a number.')
    if not form_data.get('email'):
        errors.append('Email is required.')
    elif '@' not in form_data.get('email'):
        errors.append('Invalid email address.')
    if not form_data.get('name'):
        errors.append('Name is required.')
    # ...additional validations...
    if errors:
        print(f"[ERROR] Validation errors: {errors}")
    return errors

def generate_config(form_data):
    errors = validate_form(form_data)
    if errors:
        # Log errors instead of only printing.
        print('Form validation errors:', errors)
        return None
    config = {
        'notifications': False,
        'newsletter': True,
        'preferences': form_data.get('preferences', {}),
        'age': form_data.get('age'),
        'email': form_data.get('email'),
        'name': form_data.get('name')
    }
    config_json = json.dumps(config, indent=4)
    print('Generated config JSON:', config_json)
    # Allow for batch processing merging multiple training configs.
    return config_json

# Example usage
if __name__ == '__main__':

    form_data = {
        'preferences': {},
        'age': 30,
        'email': 'john.doe@example.com',
        'name': 'John Doe'
    }
    generate_config(form_data)