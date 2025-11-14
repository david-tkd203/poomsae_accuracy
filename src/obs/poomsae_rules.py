POOMSAE_RULES = [
    "Mantener postura y equilibrio durante toda la rutina.",
    "Extensión completa de brazos y piernas en cada movimiento.",
    "Alineación correcta de caderas, hombros y cabeza.",
    "Marcación clara de inicio y final de la rutina.",
    "Seguir el reglamento oficial de la Federación Mundial de Taekwondo."
]

def get_rules_text():
    return "\n- " + "\n- ".join(POOMSAE_RULES)
