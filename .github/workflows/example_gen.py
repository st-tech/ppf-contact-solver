import re
import os

def generate_new_file(template_path, name, output_path):
    with open(template_path, 'r') as file:
        template_content = file.read()
    new_content = template_content.replace('****', name)
    with open(output_path, 'w') as file:
        file.write(new_content)

if __name__ == "__main__":
    template = 'template/example_template.yml'
    for file in os.listdir('.'):
        if re.match(r'^example_.*\.yml$', file):
            name = re.search(r'example_(.*)\.yml', file).group(1)
            output = f'example_{name}.yml'
            print(f"Generating file for: {name}")
            generate_new_file(template, name, output)
