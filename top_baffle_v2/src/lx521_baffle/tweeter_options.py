"""Public tweeter families, separate from the Stock/Slim/Obiwan baffle families."""

ND25FW = 'dayton_nd25fw4'
BMR = 'tectonic_tebm35c10_4'
ND25FN = 'dayton_nd25fn4'

TWEETER_FAMILIES = (
    dict(id=ND25FW, label='Dayton ND25FW-4 face-to-face', driver='Dayton Audio ND25FW-4',
         construction='Two domes with factory waveguide faceplates clamping the crescent',
         products=['stock', 'slim', 'obiwan']),
    dict(id=BMR, label='Tectonic TEBM35C10-4 BMR', driver='Tectonic TEBM35C10-4',
         construction='Two BMR drivers in an opposed vase or a coaxial/opposed crescent',
         products=['stock', 'slim', 'obiwan']),
    dict(id=ND25FN, label='Dayton ND25FN-4 waveguide', driver='Dayton Audio ND25FN-4',
         construction='Two faceplate-free domes in printed front/rear waveguides, fused with the MU10 UM carrier',
         products=['obiwan']),
)


def tweeter_label(identifier):
    return next(row['label'] for row in TWEETER_FAMILIES if row['id'] == identifier)


def compatible_tweeters(name, family, role):
    if family == ND25FN:
        return [ND25FN]
    if 'bmr' in name.lower():
        return [BMR]
    # Stock/Slim BMR lands have different magnet seats and no supplied
    # matching perimeter. The standard shoulders/wings are ND25FW-only.
    if role == 'regular_tweeter' or (family in ('stock', 'slim') and role in ('um', 'regular_wing')):
        return [ND25FW]
    if family == 'obiwan' and role in ('lm_bottom', 'lm_top'):
        return [ND25FW, BMR, ND25FN]
    return [ND25FW, BMR]
