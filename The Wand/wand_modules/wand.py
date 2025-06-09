import os, json, importlib

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'wand_config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    config = load_config()
    # Loop through configured modules to execute them in order.
    for module_conf in config.get("modules", []):
        module_name = module_conf.get("name")
        module_config = module_conf.get("config", {})
        try:
            mod = importlib.import_module(f"wand_modules.{module_name}")
            if hasattr(mod, "execute"):
                mod.execute(module_config)
            else:
                print(f"Module {module_name} has no execute() function.")
        except Exception as e:
            print(f"Error executing module {module_name}: {e}")

if __name__ == "__main__":
    main()
