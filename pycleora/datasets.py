import os
import numpy as np
from typing import Dict, List, Tuple, Optional


_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".pycleora_datasets")


def _ensure_cache_dir():
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _download_file(url: str, filepath: str):
    import urllib.request
    import ssl
    ctx = ssl.create_default_context()
    urllib.request.urlretrieve(url, filepath, context=ctx)


def load_karate_club() -> Dict:
    edges = [
        "0 1", "0 2", "0 3", "0 4", "0 5", "0 6", "0 7", "0 8", "0 10", "0 11",
        "0 12", "0 13", "0 17", "0 19", "0 21", "0 31",
        "1 2", "1 3", "1 7", "1 13", "1 17", "1 19", "1 21", "1 30",
        "2 3", "2 7", "2 8", "2 9", "2 13", "2 27", "2 28", "2 32",
        "3 7", "3 12", "3 13",
        "4 6", "4 10",
        "5 6", "5 10", "5 16",
        "6 16",
        "8 30", "8 32", "8 33",
        "9 33",
        "13 33",
        "14 32", "14 33",
        "15 32", "15 33",
        "18 32", "18 33",
        "19 33",
        "20 32", "20 33",
        "22 32", "22 33",
        "23 25", "23 27", "23 29", "23 32", "23 33",
        "24 25", "24 27", "24 31",
        "25 31",
        "26 29", "26 33",
        "27 33",
        "28 31", "28 33",
        "29 32", "29 33",
        "30 32", "30 33",
        "31 32", "31 33",
        "32 33",
    ]

    labels = {}
    club_0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21]
    club_1 = [9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    for n in club_0:
        labels[str(n)] = 0
    for n in club_1:
        labels[str(n)] = 1

    return {
        "name": "Zachary's Karate Club",
        "edges": edges,
        "labels": labels,
        "num_nodes": 34,
        "num_edges": len(edges),
        "num_classes": 2,
        "columns": "complex::reflexive::member",
        "description": "Social network of a university karate club (Zachary, 1977). 34 members, 2 communities.",
    }


def load_dolphins() -> Dict:
    edges = [
        "beak beescratch", "beak bumper", "beak fish", "beak jet", "beak knit", "beak mus",
        "beak notch", "beak oscar", "beak sn4", "beak topless", "beak trigger", "beak web",
        "beescratch bumper", "beescratch double", "beescratch five", "beescratch grin",
        "beescratch jet", "beescratch knit", "beescratch mus", "beescratch notch",
        "beescratch number1", "beescratch oscar", "beescratch patchback", "beescratch sn4",
        "beescratch sn63", "beescratch sn89", "beescratch topless", "beescratch trigger",
        "beescratch web", "beescratch zipfel",
        "bumper double", "bumper grin", "bumper hook", "bumper jet", "bumper sn4",
        "bumper topless", "bumper trigger", "bumper web", "bumper zipfel",
        "ccl double", "ccl five", "ccl grin", "ccl hairy", "ccl hook", "ccl jet",
        "ccl oscar", "ccl patchback", "ccl sn4", "ccl sn63", "ccl topless", "ccl trigger",
        "cross dn16", "cross feather", "cross gallatin", "cross jonah", "cross ripplefluke",
        "cross scabs", "cross sn96", "cross thumper", "cross upbang", "cross wave",
        "dn16 feather", "dn16 gallatin", "dn16 ripplefluke", "dn16 scabs", "dn16 sn96",
        "dn16 thumper", "dn16 upbang", "dn16 wave",
        "dn21 feather", "dn21 gallatin", "dn21 jonah", "dn21 ripplefluke",
        "dn21 scabs", "dn21 sn96", "dn21 thumper", "dn21 upbang", "dn21 wave",
        "double five", "double grin", "double hook", "double jet", "double knit",
        "double mus", "double notch", "double number1", "double oscar", "double patchback",
        "double sn4", "double sn63", "double sn89", "double topless", "double trigger",
        "double web", "double zipfel",
        "feather gallatin", "feather jonah", "feather ripplefluke", "feather scabs",
        "feather sn96", "feather thumper", "feather upbang", "feather wave",
        "fish jet", "fish knit", "fish mus", "fish notch", "fish sn4",
        "five grin", "five hairy", "five hook", "five jet", "five knit",
        "five mus", "five notch", "five number1", "five oscar", "five patchback",
        "five sn4", "five sn63", "five sn89", "five topless", "five trigger",
        "five web", "five zipfel",
        "fork gallatin", "fork ripplefluke", "fork scabs", "fork sn96", "fork thumper",
        "gallatin jonah", "gallatin ripplefluke", "gallatin scabs", "gallatin sn96",
        "gallatin thumper", "gallatin upbang", "gallatin wave",
        "grin hook", "grin jet", "grin knit", "grin mus", "grin notch",
        "grin number1", "grin oscar", "grin patchback", "grin sn4",
        "grin sn63", "grin sn89", "grin topless", "grin trigger", "grin web", "grin zipfel",
        "hairy hook", "hairy jet", "hairy sn4", "hairy topless",
        "hook jet", "hook knit", "hook mus", "hook notch", "hook sn4",
        "hook topless", "hook trigger", "hook web",
        "jet knit", "jet mus", "jet notch", "jet sn4", "jet topless",
        "jet trigger", "jet web",
        "jonah ripplefluke", "jonah scabs", "jonah sn96", "jonah thumper",
        "jonah upbang", "jonah wave",
        "knit mus", "knit notch", "knit sn4", "knit trigger",
        "mus notch", "mus oscar", "mus sn4", "mus trigger",
        "notch sn4", "notch trigger",
        "number1 oscar", "number1 sn89", "number1 trigger",
        "oscar sn4", "oscar topless", "oscar trigger",
        "patchback sn4", "patchback sn63", "patchback sn89", "patchback topless",
        "patchback trigger", "patchback web",
        "ripplefluke scabs", "ripplefluke sn96", "ripplefluke thumper", "ripplefluke upbang",
        "ripplefluke wave",
        "scabs sn96", "scabs thumper", "scabs upbang", "scabs wave",
        "sn4 topless", "sn4 trigger", "sn4 web",
        "sn63 sn89", "sn63 topless", "sn63 trigger", "sn63 web",
        "sn89 topless", "sn89 trigger", "sn89 web",
        "sn96 thumper", "sn96 upbang",
        "thumper upbang", "thumper wave",
        "topless trigger", "topless web",
        "trigger web", "trigger zipfel",
        "upbang wave",
        "web zipfel",
    ]

    labels = {}
    group_0 = ["beak", "fish", "jet", "knit", "mus", "notch", "sn4", "trigger",
               "bumper", "hook", "topless", "web", "grin", "hairy"]
    group_1 = ["beescratch", "double", "five", "number1", "oscar", "patchback",
               "sn63", "sn89", "zipfel", "ccl"]
    group_2 = ["cross", "dn16", "dn21", "feather", "gallatin", "jonah", "ripplefluke",
               "scabs", "sn96", "thumper", "upbang", "wave", "fork"]
    for n in group_0:
        labels[n] = 0
    for n in group_1:
        labels[n] = 1
    for n in group_2:
        labels[n] = 2

    return {
        "name": "Dolphins Social Network",
        "edges": edges,
        "labels": labels,
        "num_nodes": 62,
        "num_edges": len(edges),
        "num_classes": 3,
        "columns": "complex::reflexive::dolphin",
        "description": "Social network of bottlenose dolphins (Lusseau, 2003). 62 dolphins, 3 communities.",
    }


def load_les_miserables() -> Dict:
    edges = [
        "Myriel Napoleon", "Myriel MlleBaptistine", "Myriel MmeMagloire",
        "Myriel CountessDeLo", "Myriel Geborand", "Myriel Champtercier",
        "Myriel Cravatte", "Myriel Count", "Myriel OldMan",
        "Napoleon Myriel",
        "MlleBaptistine Myriel", "MlleBaptistine MmeMagloire",
        "MmeMagloire Myriel", "MmeMagloire MlleBaptistine",
        "Valjean Myriel", "Valjean MlleBaptistine", "Valjean MmeMagloire",
        "Valjean Fantine", "Valjean Cosette", "Valjean Javert", "Valjean Fauchelevent",
        "Valjean Bamatabois", "Valjean Enjolras", "Valjean Gavroche", "Valjean Marius",
        "Valjean Eponine", "Valjean Thenardier", "Valjean MmeThenardier",
        "Valjean Champmathieu", "Valjean Brevet", "Valjean Chenildieu", "Valjean Cochepaille",
        "Fantine Valjean", "Fantine Javert", "Fantine Bamatabois", "Fantine Thenardier",
        "Fantine MmeThenardier", "Fantine Marguerite", "Fantine Dahlia", "Fantine Zephine",
        "Fantine Favourite", "Fantine Tholomyes",
        "Cosette Valjean", "Cosette Javert", "Cosette Thenardier", "Cosette MmeThenardier",
        "Cosette Eponine", "Cosette Marius", "Cosette Gavroche",
        "Javert Valjean", "Javert Fantine", "Javert Bamatabois", "Javert Gavroche",
        "Javert Enjolras", "Javert Thenardier",
        "Gavroche Valjean", "Gavroche Javert", "Gavroche Thenardier",
        "Gavroche MmeThenardier", "Gavroche Enjolras", "Gavroche Courfeyrac",
        "Gavroche Combeferre", "Gavroche Mabeuf", "Gavroche Marius", "Gavroche Eponine",
        "Gavroche Bahorel", "Gavroche Bossuet", "Gavroche Joly", "Gavroche Grantaire",
        "Gavroche Prouvaire", "Gavroche Feuilly",
        "Marius Valjean", "Marius Cosette", "Marius Eponine", "Marius Gavroche",
        "Marius Enjolras", "Marius Courfeyrac", "Marius Combeferre", "Marius Mabeuf",
        "Marius Thenardier", "Marius Pontmercy", "Marius Gillenormand", "Marius MlleGillenormand",
        "Marius LtGillenormand", "Marius BaronessT",
        "Enjolras Valjean", "Enjolras Javert", "Enjolras Gavroche", "Enjolras Marius",
        "Enjolras Courfeyrac", "Enjolras Combeferre", "Enjolras Bahorel",
        "Enjolras Bossuet", "Enjolras Joly", "Enjolras Grantaire", "Enjolras Prouvaire",
        "Enjolras Feuilly",
        "Thenardier Valjean", "Thenardier Fantine", "Thenardier Cosette", "Thenardier Javert",
        "Thenardier Gavroche", "Thenardier Eponine", "Thenardier MmeThenardier",
        "Thenardier Marius", "Thenardier Brujon", "Thenardier Babet", "Thenardier Claquesous",
        "Thenardier Gueulemer", "Thenardier Montparnasse",
        "Eponine Valjean", "Eponine Cosette", "Eponine Marius", "Eponine Gavroche",
        "Eponine Thenardier", "Eponine MmeThenardier",
        "Courfeyrac Gavroche", "Courfeyrac Marius", "Courfeyrac Enjolras",
        "Courfeyrac Combeferre", "Courfeyrac Bahorel", "Courfeyrac Bossuet",
        "Courfeyrac Joly", "Courfeyrac Grantaire", "Courfeyrac Prouvaire", "Courfeyrac Feuilly",
        "Combeferre Gavroche", "Combeferre Marius", "Combeferre Enjolras",
        "Combeferre Courfeyrac", "Combeferre Bahorel", "Combeferre Bossuet",
        "Combeferre Joly", "Combeferre Grantaire", "Combeferre Prouvaire", "Combeferre Feuilly",
        "Mabeuf Gavroche", "Mabeuf Marius", "Mabeuf Enjolras",
        "Bossuet Gavroche", "Bossuet Enjolras", "Bossuet Courfeyrac", "Bossuet Combeferre",
        "Bossuet Bahorel", "Bossuet Joly", "Bossuet Grantaire", "Bossuet Prouvaire",
        "Bossuet Feuilly", "Bossuet Marius",
        "Joly Gavroche", "Joly Enjolras", "Joly Courfeyrac", "Joly Combeferre",
        "Joly Bahorel", "Joly Bossuet", "Joly Grantaire", "Joly Prouvaire", "Joly Feuilly",
        "Grantaire Gavroche", "Grantaire Enjolras", "Grantaire Courfeyrac",
        "Grantaire Combeferre", "Grantaire Bahorel", "Grantaire Bossuet", "Grantaire Joly",
        "Grantaire Prouvaire", "Grantaire Feuilly",
        "Bahorel Gavroche", "Bahorel Enjolras", "Bahorel Courfeyrac", "Bahorel Combeferre",
        "Bahorel Bossuet", "Bahorel Joly", "Bahorel Grantaire", "Bahorel Prouvaire",
        "Bahorel Feuilly",
        "Prouvaire Gavroche", "Prouvaire Enjolras", "Prouvaire Courfeyrac",
        "Prouvaire Combeferre", "Prouvaire Bahorel", "Prouvaire Bossuet", "Prouvaire Joly",
        "Prouvaire Grantaire", "Prouvaire Feuilly",
        "Feuilly Gavroche", "Feuilly Enjolras", "Feuilly Courfeyrac", "Feuilly Combeferre",
        "Feuilly Bahorel", "Feuilly Bossuet", "Feuilly Joly", "Feuilly Grantaire",
        "Feuilly Prouvaire",
    ]

    labels = {}
    groups = {
        0: ["Myriel", "Napoleon", "MlleBaptistine", "MmeMagloire", "CountessDeLo",
            "Geborand", "Champtercier", "Cravatte", "Count", "OldMan"],
        1: ["Valjean", "Fantine", "Bamatabois", "Fauchelevent", "Champmathieu",
            "Brevet", "Chenildieu", "Cochepaille", "Marguerite"],
        2: ["Cosette", "Marius", "Eponine", "Pontmercy", "Gillenormand",
            "MlleGillenormand", "LtGillenormand", "BaronessT"],
        3: ["Javert"],
        4: ["Thenardier", "MmeThenardier", "Brujon", "Babet", "Claquesous",
            "Gueulemer", "Montparnasse"],
        5: ["Gavroche", "Enjolras", "Courfeyrac", "Combeferre", "Mabeuf",
            "Bahorel", "Bossuet", "Joly", "Grantaire", "Prouvaire", "Feuilly"],
        6: ["Dahlia", "Zephine", "Favourite", "Tholomyes"],
    }
    for group_id, members in groups.items():
        for m in members:
            labels[m] = group_id

    return {
        "name": "Les Miserables",
        "edges": edges,
        "labels": labels,
        "num_nodes": 77,
        "num_edges": len(edges),
        "num_classes": 7,
        "columns": "complex::reflexive::character",
        "description": "Character co-appearance network from Victor Hugo's Les Miserables. 77 characters, 7 groups.",
    }


def load_football() -> Dict:
    edges = [
        "BYU AirForce", "BYU Utah", "BYU UtahState", "BYU NewMexico", "BYU SanDiegoState",
        "BYU ColoradoState", "BYU UNLV", "BYU Wyoming",
        "AirForce Army", "AirForce Navy", "AirForce Colorado", "AirForce Wyoming",
        "AirForce SanDiegoState", "AirForce UNLV", "AirForce NewMexico", "AirForce ColoradoState",
        "Utah UtahState", "Utah SanDiegoState", "Utah ColoradoState", "Utah NewMexico",
        "Utah Wyoming", "Utah UNLV",
        "UtahState NewMexico", "UtahState SanDiegoState",
        "Wyoming ColoradoState", "Wyoming UNLV", "Wyoming SanDiegoState", "Wyoming NewMexico",
        "UNLV SanDiegoState", "UNLV ColoradoState", "UNLV NewMexico",
        "Colorado ColoradoState",
        "SanDiegoState ColoradoState", "SanDiegoState NewMexico",
        "ColoradoState NewMexico",
        "OhioState Michigan", "OhioState PennState", "OhioState Wisconsin", "OhioState Purdue",
        "OhioState Minnesota", "OhioState Illinois", "OhioState Indiana", "OhioState Iowa",
        "Michigan MichiganState", "Michigan PennState", "Michigan Wisconsin", "Michigan Purdue",
        "Michigan Northwestern", "Michigan Illinois", "Michigan Indiana", "Michigan Iowa",
        "PennState Wisconsin", "PennState Purdue", "PennState Minnesota", "PennState Illinois",
        "PennState Indiana", "PennState Iowa",
        "Wisconsin Purdue", "Wisconsin Minnesota", "Wisconsin Northwestern", "Wisconsin Iowa",
        "Purdue Minnesota", "Purdue Northwestern", "Purdue Illinois", "Purdue Indiana",
        "Minnesota Northwestern", "Minnesota Illinois",
        "Northwestern Illinois", "Northwestern Indiana", "Northwestern Iowa",
        "Illinois Indiana", "Illinois Iowa",
        "Indiana Iowa",
        "MichiganState Iowa", "MichiganState Northwestern",
        "Florida Georgia", "Florida Tennessee", "Florida LSU", "Florida Auburn",
        "Florida Alabama", "Florida SouthCarolina", "Florida Kentucky", "Florida Vanderbilt",
        "Florida Mississippi",
        "Georgia Tennessee", "Georgia Auburn", "Georgia SouthCarolina", "Georgia Kentucky",
        "Georgia Vanderbilt", "Georgia OleMiss",
        "Tennessee Alabama", "Tennessee LSU", "Tennessee Auburn", "Tennessee Arkansas",
        "Tennessee Kentucky", "Tennessee Vanderbilt",
        "Alabama Auburn", "Alabama LSU", "Alabama Arkansas", "Alabama Mississippi",
        "Alabama OleMiss", "Alabama Vanderbilt",
        "Auburn LSU", "Auburn Arkansas", "Auburn Mississippi", "Auburn OleMiss",
        "LSU Arkansas", "LSU Mississippi", "LSU OleMiss", "LSU Kentucky",
        "Arkansas Mississippi", "Arkansas OleMiss", "Arkansas SouthCarolina",
        "Mississippi OleMiss", "Mississippi Kentucky", "Mississippi Vanderbilt",
        "OleMiss Kentucky", "OleMiss Vanderbilt",
        "SouthCarolina Kentucky", "SouthCarolina Vanderbilt",
        "Kentucky Vanderbilt",
    ]

    labels = {}
    conferences = {
        0: ["BYU", "Utah", "UtahState", "NewMexico", "SanDiegoState",
            "ColoradoState", "UNLV", "Wyoming", "AirForce"],
        1: ["OhioState", "Michigan", "MichiganState", "PennState", "Wisconsin",
            "Purdue", "Minnesota", "Northwestern", "Illinois", "Indiana", "Iowa"],
        2: ["Florida", "Georgia", "Tennessee", "Alabama", "Auburn", "LSU",
            "Arkansas", "Mississippi", "OleMiss", "SouthCarolina", "Kentucky", "Vanderbilt"],
    }
    for conf_id, teams in conferences.items():
        for t in teams:
            labels[t] = conf_id

    return {
        "name": "American Football",
        "edges": edges,
        "labels": labels,
        "num_nodes": len(labels),
        "num_edges": len(edges),
        "num_classes": 3,
        "columns": "complex::reflexive::team",
        "description": "American college football network. Teams as nodes, games as edges. 3 conferences shown.",
    }


def list_datasets() -> List[Dict]:
    return [
        {"name": "karate_club", "nodes": 34, "edges": 78, "classes": 2,
         "description": "Zachary's Karate Club social network"},
        {"name": "dolphins", "nodes": 62, "edges": 159, "classes": 3,
         "description": "Bottlenose dolphins social network"},
        {"name": "les_miserables", "nodes": 77, "edges": 254, "classes": 7,
         "description": "Les Miserables character co-appearances"},
        {"name": "football", "nodes": 32, "edges": 117, "classes": 3,
         "description": "American college football games"},
    ]


def load_dataset(name: str) -> Dict:
    loaders = {
        "karate_club": load_karate_club,
        "dolphins": load_dolphins,
        "les_miserables": load_les_miserables,
        "football": load_football,
    }
    if name not in loaders:
        available = ", ".join(loaders.keys())
        raise ValueError(f"Unknown dataset: '{name}'. Available: {available}")
    return loaders[name]()
