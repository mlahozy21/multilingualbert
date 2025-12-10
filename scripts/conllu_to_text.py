import sys

def convert(conllu_path, output_path):
    print(f"Limpiando {conllu_path} -> {output_path} ...")
    with open(conllu_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        tokens = []
        for line in f_in:
            line = line.strip()
            if not line:
                if tokens:
                    f_out.write(" ".join(tokens) + "\n")
                    tokens = []
            elif line.startswith("#"):
                continue
            else:
                parts = line.split("\t")
                # CORRECCIÓN: Verificar que sea una palabra real y no un rango
                # Las líneas de rango tienen un guion en el ID (ej: "1-2")
                if len(parts) > 0 and "-" not in parts[0]:
                    if len(parts) > 1:
                        tokens.append(parts[1])
        
        if tokens: 
            f_out.write(" ".join(tokens) + "\n")
    print("¡Hecho!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python conllu_to_text.py <entrada.conllu> <salida.txt>")
    else:
        convert(sys.argv[1], sys.argv[2])