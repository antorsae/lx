"""Public tweeter families, separate from the Stock/Slim/Obiwan baffle families."""

ND25FW = 'dayton_nd25fw4'
BMR = 'tectonic_tebm35c10_4'
ND25FN = 'dayton_nd25fn4'
PT2522 = 'grs_pt2522_4'
AMT17 = 'mundorf_amt17d2_1'
DIPOLE = (PT2522, AMT17)

TWEETER_FAMILIES = (
    dict(id=ND25FW, label='Dayton ND25FW-4 face-to-face', driver='Dayton Audio ND25FW-4',
         construction='Two domes with factory waveguide faceplates clamping the crescent',
         products=['stock', 'slim', 'obiwan']),
    dict(id=BMR, label='Tectonic TEBM35C10-4 BMR', driver='Tectonic TEBM35C10-4',
         construction='Two BMR drivers in an opposed vase or a coaxial/opposed crescent',
         products=['stock', 'slim', 'obiwan']),
    dict(id=ND25FN, label='Dayton ND25FN-4 waveguide', driver='Dayton Audio ND25FN-4',
         construction='Two faceplate-free domes in printed front/rear waveguides, fused with the MU10 upper',
         products=['stock', 'obiwan']),
    dict(id=PT2522, label='GRS PT2522-4 dipole planar', driver='GRS PT2522-4',
         construction='One open-back planar on a frame above (T above) or below (T below) the raised MU10, '
                      'upright or rotated, M3 heat-set inserts',
         products=['obiwan']),
    dict(id=AMT17, label='Mundorf AMT17D2.1 dipole AMT', driver='Mundorf AMT17D2.1',
         construction='One open-back AMT on a frame above (T above) or below (T below) the raised MU10, '
                      'upright or rotated, M3 heat-set inserts',
         products=['obiwan']),
)


def tweeter_label(identifier):
    return next(row['label'] for row in TWEETER_FAMILIES if row['id'] == identifier)


def compatible_tweeters(name, family, role):
    if family in (ND25FN, *DIPOLE):
        return [family]
    if 'bmr' in name.lower():
        return [BMR]
    # Stock/Slim BMR lands have different magnet seats and no supplied
    # matching perimeter. The standard shoulders/wings are ND25FW-only.
    if role == 'regular_tweeter' or (family in ('stock', 'slim') and role in ('um', 'regular_wing')):
        return [ND25FW]
    # The Stock ND25FN-4 vase replaces the upper module on the H2C Stock LM.
    # The Obiwan ND25FN-4 vase top replaces the keyed LM top, so it keys onto
    # the keyed LM bottom; the one-piece Obiwan LM carrier does not take it.
    # The dipole parts (Obiwan only): T above replaces the UM carrier on the one-piece LM carrier or the keyed LM;
    # T below replaces the keyed LM top (one piece, or a keyed LM top variant plus a T + UM piece).
    # The NL8 service lid closes the bay of every Obiwan floor-stand LM, the keyed bottom under the vase top included.
    dipole = list(DIPOLE) if family == 'obiwan' and role in ('lm_bottom', 'lm_top') else []
    if (family == 'stock' and role in ('lm_bottom', 'lm_top')) or 'lm_keyed_bottom' in name or 'nl8_service_lid' in name:
        return [ND25FW, BMR, ND25FN] + dipole
    return [ND25FW, BMR] + dipole
