"""Seed channels for dataset collection, keyed by content type.

Owned by the collection pair. Handles rather than channel IDs: the API resolves
them, so there are no hand-copied IDs to get wrong, and a handle that fails to
resolve is reported and skipped rather than silently collecting nothing.

Verified against the API on 2026-09-16: every handle below resolved to a real
channel above MIN_SUBSCRIBERS. Handle squatters exist (@Noclip resolves to a
1,790-sub impostor, not the documentary channel), so re-verify with
`python -m src.collection verify` before adding to this list.

Keys should match config.schema.CONTENT_TYPES. A key that does not produces a
warning, not an error, so the list can be developed incrementally.

The sample is the seed list: every bias here (language, region, production
budget, the team's own viewing habits) is a bias in the dataset. Spread each
content type across channel sizes deliberately.
"""

CHANNELS = {
    "gaming": [
        "@markiplier",                       # 38,900,000
        "@jacksepticeye",                    # 31,200,000
        "@DanTDM",                           # 29,100,000
        "@VanossGaming",                     # 26,000,000
        "@SSundee",                          # 25,500,000
        "@Aphmau",                           # 25,300,000
        "@LazarBeam",                        # 23,300,000
        "@penguinz0",                        # 18,300,000
        "@videogamedunkey",                  # 7,570,000
        "@CouRageJD",                        # 4,650,000
        "@CallMeKevin",                      # 3,660,000
        "@RTGame",                           # 2,960,000
        "@Sykkuno",                          # 2,550,000
        "@NakeyJakey",                       # 2,120,000
        "@GMTK",                             # 1,740,000
        "@SsethTzeentach",                   # 1,620,000
        "@MandaloreGaming",                  # 996,000
        "@Raycevick",                        # 651,000
        "@Whitelight",                       # 564,000
        "@Civvie11",                         # 493,000
        "@Matthewmatosis",                   # 245,000
        "@ThorHighHeels",                    # 161,000
    ],
    "cooking": [
        "@JoshuaWeissman",                   # 10,700,000
        "@bingingwithbabish",                # 10,500,000
        "@BonAppetit",                       # 7,400,000
        "@Epicurious",                       # 6,310,000
        "@JunsKitchen",                      # 5,170,000
        "@FutureCanoe",                      # 4,860,000
        "@FoodWishes",                       # 4,700,000
        "@MythicalKitchen",                  # 4,550,000
        "@TastingHistory",                   # 4,470,000
        "@SamTheCookingGuy",                 # 3,860,000
        "@AmericasTestKitchen",              # 3,020,000
        "@SortedFood",                       # 3,000,000
        "@aragusea",                         # 2,620,000
        "@ThatDudeCanCook",                  # 2,520,000
        "@ChefJeanPierre",                   # 2,470,000
        "@EthanChlebowski",                  # 2,430,000
        "@Frenchguycooking",                 # 2,070,000
        "@BrianLagerstrom",                  # 1,800,000
        "@JKenjiLopezAlt",                   # 1,760,000
        "@NotAnotherCookingShow",            # 1,410,000
        "@ChineseCookingDemystified",        # 1,010,000
        "@internetshaquille",                # 849,000
        "@CharlieAndersonCooking",           # 207,000
    ],
    "finance": [
        "@MarkTilbury",                      # 8,950,000
        "@GrahamStephan",                    # 5,180,000
        "@AndreiJikh",                       # 3,380,000
        "@EconomicsExplained",               # 2,880,000
        "@MinorityMindset",                  # 2,520,000
        "@MeetKevin",                        # 2,050,000
        "@GarysEconomics",                   # 1,660,000
        "@TheFinancialDiet",                 # 1,350,000
        "@NateOBrien",                       # 1,280,000
        "@ThePlainBagel",                    # 1,210,000
        "@WhiteBoardFinance",                # 1,030,000
        "@TheSwedishInvestor",               # 1,020,000
        "@TwoCentsPBS",                      # 826,000
        "@MoneyMacro",                       # 675,000
        "@BravosResearch",                   # 643,000
        "@BenFelixCSI",                      # 641,000
        "@JosephCarlsonShow",                # 511,000
        "@DamienTalksMoney",                 # 414,000
        "@Value-Investing",                  # 270,000
    ],
    "education": [
        "@markrober",                        # 82,600,000
        "@kurzgesagt",                       # 25,600,000
        "@Vsauce",                           # 25,100,000
        "@TEDEd",                            # 22,900,000
        "@veritasium",                       # 21,200,000
        "@crashcourse",                      # 17,200,000
        "@smartereveryday",                  # 11,900,000
        "@NileRed",                          # 10,900,000
        "@ElectroBOOM",                      # 8,820,000
        "@3blue1brown",                      # 8,620,000
        "@SciShow",                          # 8,410,000
        "@MinutePhysics",                    # 5,980,000
        "@TheActionLab",                     # 5,160,000
        "@PracticalEngineeringChannel",      # 4,830,000
        "@numberphile",                      # 4,770,000
        "@tierzoo",                          # 3,940,000
        "@SteveMould",                       # 3,610,000
        "@BranchEducation",                  # 2,750,000
        "@Computerphile",                    # 2,640,000
        "@RealScience",                      # 2,090,000
        "@standupmaths",                     # 1,350,000
        "@Mathologer",                       # 968,000
        "@AppliedScience",                   # 889,000
        "@AlphaPhoenixChannel",              # 723,000
        "@Artemkirsanov",                    # 374,000
        "@Reducible",                        # 337,000
    ],
}
