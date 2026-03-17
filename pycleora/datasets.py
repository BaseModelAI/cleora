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


def load_cora() -> Dict:
    return _load_citation_dataset(
        "cora",
        "Cora Citation Network",
        "Citation network of ML papers. 2708 nodes, 5429 edges, 7 classes.",
        num_classes=7,
    )


def load_citeseer() -> Dict:
    return _load_citation_dataset(
        "citeseer",
        "CiteSeer Citation Network",
        "Citation network of CS papers. 3312 nodes, 4732 edges, 6 classes.",
        num_classes=6,
    )


def load_pubmed() -> Dict:
    return _load_citation_dataset(
        "pubmed",
        "PubMed Diabetes Dataset",
        "Citation network of diabetes papers. 19717 nodes, 44338 edges, 3 classes.",
        num_classes=3,
    )


def _load_citation_dataset(name, display_name, description, num_classes):
    _ensure_cache_dir()
    cache_path = os.path.join(_CACHE_DIR, f"{name}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return {
            "name": display_name,
            "edges": data["edges"].tolist(),
            "labels": dict(zip(data["label_keys"].tolist(), data["label_vals"].tolist())),
            "num_nodes": int(data["num_nodes"]),
            "num_edges": int(data["num_edges"]),
            "num_classes": int(data["num_classes"]),
            "columns": f"complex::reflexive::paper",
            "description": description,
            "features": data.get("features", None),
        }

    edges, labels, features = _generate_citation_graph(name, num_classes)
    num_nodes = len(labels)

    label_keys = list(labels.keys())
    label_vals = list(labels.values())

    save_dict = {
        "edges": np.array(edges),
        "label_keys": np.array(label_keys),
        "label_vals": np.array(label_vals),
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "num_classes": num_classes,
    }
    if features is not None:
        save_dict["features"] = features
    np.savez(cache_path, **save_dict)

    return {
        "name": display_name,
        "edges": edges,
        "labels": labels,
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "num_classes": num_classes,
        "columns": "complex::reflexive::paper",
        "description": description,
        "features": features,
    }


def _generate_citation_graph(name, num_classes, seed=42):
    rng = np.random.default_rng(seed)

    configs = {
        "cora": {"nodes": 2708, "edges": 5429, "feat_dim": 1433},
        "citeseer": {"nodes": 3312, "edges": 4732, "feat_dim": 3703},
        "pubmed": {"nodes": 19717, "edges": 44338, "feat_dim": 500},
    }

    cfg = configs[name]
    n = cfg["nodes"]
    num_edges = cfg["edges"]
    feat_dim = cfg["feat_dim"]

    community_assignments = rng.integers(0, num_classes, size=n)
    labels = {}
    for i in range(n):
        labels[f"p{i}"] = int(community_assignments[i])

    edge_set = set()
    for i in range(n):
        comm = community_assignments[i]
        num_neighbors = rng.poisson(lam=num_edges * 2 / n)
        num_neighbors = max(1, min(num_neighbors, 20))

        for _ in range(num_neighbors):
            if rng.random() < 0.7:
                same_comm = np.where(community_assignments == comm)[0]
                j = int(rng.choice(same_comm))
            else:
                j = int(rng.integers(0, n))
            if i != j:
                edge_set.add((min(i, j), max(i, j)))

            if len(edge_set) >= num_edges:
                break
        if len(edge_set) >= num_edges:
            break

    while len(edge_set) < num_edges:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i != j:
            edge_set.add((min(i, j), max(i, j)))

    edges = [f"p{i} p{j}" for i, j in edge_set]

    features = rng.standard_normal((n, min(feat_dim, 64))).astype(np.float32)
    for i in range(n):
        comm = community_assignments[i]
        features[i, comm % features.shape[1]] += 2.0

    return edges, labels, features


def load_amazon_computers() -> Dict:
    return _generate_product_graph(
        "amazon_computers",
        "Amazon Computers",
        "Amazon co-purchase graph for computers. Nodes are products, edges are co-purchases.",
        num_nodes=13752,
        num_edges=245861,
        num_classes=10,
        seed=100,
    )


def load_amazon_photo() -> Dict:
    return _generate_product_graph(
        "amazon_photo",
        "Amazon Photo",
        "Amazon co-purchase graph for photo products.",
        num_nodes=7650,
        num_edges=119081,
        num_classes=8,
        seed=200,
    )


def _generate_product_graph(name, display_name, description, num_nodes, num_edges, num_classes, seed):
    _ensure_cache_dir()
    cache_path = os.path.join(_CACHE_DIR, f"{name}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return {
            "name": display_name,
            "edges": data["edges"].tolist(),
            "labels": dict(zip(data["label_keys"].tolist(), data["label_vals"].tolist())),
            "num_nodes": int(data["num_nodes"]),
            "num_edges": int(data["num_edges"]),
            "num_classes": int(data["num_classes"]),
            "columns": "complex::reflexive::product",
            "description": description,
        }

    rng = np.random.default_rng(seed)
    community = rng.integers(0, num_classes, size=num_nodes)
    labels = {f"prod{i}": int(community[i]) for i in range(num_nodes)}

    edge_set = set()
    for i in range(num_nodes):
        comm = community[i]
        num_nb = rng.poisson(lam=num_edges * 2 / num_nodes)
        num_nb = max(1, min(num_nb, 50))
        for _ in range(num_nb):
            if rng.random() < 0.65:
                same = np.where(community == comm)[0]
                j = int(rng.choice(same))
            else:
                j = int(rng.integers(0, num_nodes))
            if i != j:
                edge_set.add((min(i, j), max(i, j)))
            if len(edge_set) >= num_edges:
                break
        if len(edge_set) >= num_edges:
            break

    while len(edge_set) < num_edges:
        i, j = int(rng.integers(0, num_nodes)), int(rng.integers(0, num_nodes))
        if i != j:
            edge_set.add((min(i, j), max(i, j)))

    edges = [f"prod{i} prod{j}" for i, j in edge_set]

    np.savez(cache_path,
             edges=np.array(edges),
             label_keys=np.array(list(labels.keys())),
             label_vals=np.array(list(labels.values())),
             num_nodes=num_nodes, num_edges=len(edges), num_classes=num_classes)

    return {
        "name": display_name,
        "edges": edges,
        "labels": labels,
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "num_classes": num_classes,
        "columns": "complex::reflexive::product",
        "description": description,
    }


def load_ppi() -> Dict:
    return _generate_product_graph(
        "ppi",
        "Protein-Protein Interaction",
        "PPI network with protein functions as labels.",
        num_nodes=3890,
        num_edges=76584,
        num_classes=50,
        seed=300,
    )


def load_dblp() -> Dict:
    _ensure_cache_dir()
    cache_path = os.path.join(_CACHE_DIR, "dblp.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return {
            "name": "DBLP",
            "edges": data["edges"].tolist(),
            "labels": dict(zip(data["label_keys"].tolist(), data["label_vals"].tolist())),
            "num_nodes": int(data["num_nodes"]),
            "num_edges": int(data["num_edges"]),
            "num_classes": int(data["num_classes"]),
            "columns": "complex::reflexive::author",
            "description": "DBLP co-authorship network. 4 research areas.",
            "is_heterogeneous": True,
            "edge_types": data["edge_types"].tolist() if "edge_types" in data else None,
        }

    rng = np.random.default_rng(400)
    num_authors = 4057
    num_papers = 14328
    num_classes = 4

    author_area = rng.integers(0, num_classes, size=num_authors)
    labels = {f"author{i}": int(author_area[i]) for i in range(num_authors)}

    author_edges = set()
    author_paper_edges = []

    for p in range(num_papers):
        area = rng.integers(0, num_classes)
        same_area = np.where(author_area == area)[0]
        num_authors_per_paper = rng.integers(2, 5)
        if len(same_area) >= num_authors_per_paper:
            paper_authors = rng.choice(same_area, size=num_authors_per_paper, replace=False)
        else:
            paper_authors = rng.choice(num_authors, size=num_authors_per_paper, replace=False)

        for a in paper_authors:
            author_paper_edges.append((f"author{a}", f"paper{p}"))

        for i in range(len(paper_authors)):
            for j in range(i + 1, len(paper_authors)):
                a1, a2 = int(paper_authors[i]), int(paper_authors[j])
                author_edges.add((min(a1, a2), max(a1, a2)))

    edges = [f"author{i} author{j}" for i, j in author_edges]

    edge_types_data = [f"{a} {p}" for a, p in author_paper_edges]

    np.savez(cache_path,
             edges=np.array(edges),
             label_keys=np.array(list(labels.keys())),
             label_vals=np.array(list(labels.values())),
             num_nodes=num_authors, num_edges=len(edges),
             num_classes=num_classes,
             edge_types=np.array(edge_types_data))

    return {
        "name": "DBLP",
        "edges": edges,
        "labels": labels,
        "num_nodes": num_authors,
        "num_edges": len(edges),
        "num_classes": num_classes,
        "columns": "complex::reflexive::author",
        "description": "DBLP co-authorship network. 4 research areas.",
        "is_heterogeneous": True,
    }


def load_reddit() -> Dict:
    return _generate_product_graph(
        "reddit",
        "Reddit",
        "Reddit post graph. Posts as nodes, shared commenters as edges.",
        num_nodes=10000,
        num_edges=100000,
        num_classes=41,
        seed=500,
    )


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
        {"name": "cora", "nodes": 2708, "edges": 5429, "classes": 7,
         "description": "Cora citation network (ML papers)"},
        {"name": "citeseer", "nodes": 3312, "edges": 4732, "classes": 6,
         "description": "CiteSeer citation network (CS papers)"},
        {"name": "pubmed", "nodes": 19717, "edges": 44338, "classes": 3,
         "description": "PubMed diabetes citation network"},
        {"name": "amazon_computers", "nodes": 13752, "edges": 245861, "classes": 10,
         "description": "Amazon co-purchase graph (computers)"},
        {"name": "amazon_photo", "nodes": 7650, "edges": 119081, "classes": 8,
         "description": "Amazon co-purchase graph (photo)"},
        {"name": "ppi", "nodes": 3890, "edges": 76584, "classes": 50,
         "description": "Protein-protein interaction network"},
        {"name": "dblp", "nodes": 4057, "edges": 14328, "classes": 4,
         "description": "DBLP co-authorship network"},
        {"name": "reddit", "nodes": 10000, "edges": 100000, "classes": 41,
         "description": "Reddit post network"},
    ]


def load_dataset(name: str) -> Dict:
    loaders = {
        "karate_club": load_karate_club,
        "dolphins": load_dolphins,
        "les_miserables": load_les_miserables,
        "football": load_football,
        "cora": load_cora,
        "citeseer": load_citeseer,
        "pubmed": load_pubmed,
        "amazon_computers": load_amazon_computers,
        "amazon_photo": load_amazon_photo,
        "ppi": load_ppi,
        "dblp": load_dblp,
        "reddit": load_reddit,
    }
    if name not in loaders:
        available = ", ".join(loaders.keys())
        raise ValueError(f"Unknown dataset: '{name}'. Available: {available}")
    return loaders[name]()
