"""ASCII maps of the Clean Up game.

Legend (from the SSD sources): ``@`` wall, ``P`` player spawn point, ``B`` apple spawn cell, ``H`` waste
spawn cell (starts polluted), ``R`` river cell, ``S`` stream cell, `` `` empty floor.

``CLEANUP_MAP`` is the original 5-player map of Hughes et al. (2018) / Vinitsky et al.; ``CLEANUP_10x10_SYM``
and ``CLEANUP_SMALL_SYM`` are the small maps of Yang et al. (2020, "Learning to Incentivize Other Learning
Agents", https://github.com/011235813/lio, MIT licence): 3 agents on 10x10, 2 agents on 7x7.
"""

CLEANUP_MAP = [
    '@@@@@@@@@@@@@@@@@@',
    '@RRRRRR     BBBBB@',
    '@HHHHHH      BBBB@',
    '@RRRRRR     BBBBB@',
    '@RRRRR  P    BBBB@',
    '@RRRRR    P BBBBB@',
    '@HHHHH       BBBB@',
    '@RRRRR      BBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@RRRRR   P P BBBB@',
    '@HHHHH   P  BBBBB@',
    '@RRRRRR    P BBBB@',
    '@HHHHHH P   BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH    P  BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH  P P BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@@@@@@@@@@@@@@@@@@']

# LIO, 3 agents: river / waste columns on the left, apple cells on the right, three spawn points
CLEANUP_10x10_SYM = [
    '@@@@@@@@@@',
    '@HH   P B@',
    '@RR    BB@',
    '@HH     B@',
    '@RR    BB@',
    '@HH P   B@',
    '@RR    BB@',
    '@HH     B@',
    '@RRP   BB@',
    '@@@@@@@@@@']

# LIO, 2 agents: agent 0 spawns on the river side, agent 1 on the apple side
CLEANUP_SMALL_SYM = [
    '@@@@@@@',
    '@H  PB@',
    '@H   B@',
    '@    B@',
    '@    B@',
    '@ P  B@',
    '@@@@@@@']

MAPS = {"10x10": CLEANUP_10x10_SYM, "7x7": CLEANUP_SMALL_SYM, "original": CLEANUP_MAP}

# LIO's Clean Up parameters per map (Yang et al. 2020, Table 3); the original map uses the SSD defaults
CLEANUP_PARAMS = {
    "10x10": {"appleRespawnProbability": 0.3, "thresholdDepletion": 0.4, "thresholdRestoration": 0.0,
              "wasteSpawnProbability": 0.5},
    "7x7": {"appleRespawnProbability": 0.5, "thresholdDepletion": 0.6, "thresholdRestoration": 0.0,
            "wasteSpawnProbability": 0.5},
    "original": {"appleRespawnProbability": 0.05, "thresholdDepletion": 0.4, "thresholdRestoration": 0.0,
                 "wasteSpawnProbability": 0.5},
}
