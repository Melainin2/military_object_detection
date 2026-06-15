import json
import logging
import os
import urllib.request

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

logger = logging.getLogger("backend.vit")

IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

MILITARY_KEYWORDS = {
    "soldier", "troop", "infantry", "marine", "commando", "paratrooper", "gunner",
    "marksman", "sharpshooter", "sniper", "patrol", "convoy", "cavalry", "legion",
    "garrison", "battalion", "regiment", "division", "brigade", "squadron",
    "platoon", "artillery", "howitzer", "mortar", "cannon", "bazooka", "launcher",
    "machine gun", "machinegun", "rifle", "assault rifle", "carbine", "shotgun",
    "pistol", "revolver", "grenade", "landmine", "explosive", "bomb", "missile",
    "projectile", "warhead", "ammunition", "ammo", "munition",
    "tank", "armored vehicle", "armoured vehicle", "armored car", "armoured car",
    "armored personnel carrier", "armoured personnel carrier", "apc",
    "infantry fighting vehicle", "ifv", "mrap", "humvee", "jeep",
    "military truck", "supply truck", "troop carrier",
    "battle tank", "main battle tank", "light tank", "heavy tank",
    "tank destroyer", "assault gun", "self-propelled gun", "self propelled gun",
    "anti-aircraft", "anti aircraft", "anti-tank", "anti tank",
    "radar", "sonar", "periscope", "binoculars",
    "helmet", "kevlar", "flak jacket", "body armor", "body armour",
    "camouflage", "camo", "fatigues", "combat uniform", "military uniform",
    "bunker", "fortification", "fortress", "pillbox", "trench", "foxhole",
    "barricade", "stronghold", "citadel", "bastion", "outpost", "watchtower",
    "sentry", "guard post", "checkpoint",
    "warship", "battleship", "destroyer", "cruiser", "frigate", "corvette",
    "submarine", "aircraft carrier", "carrier", "landing craft",
    "patrol boat", "missile boat", "torpedo boat", "navy", "naval",
    "fighter jet", "fighter aircraft", "fighter plane", "fighter",
    "bomber", "stealth bomber", "bomber aircraft",
    "attack aircraft", "warplane", "military plane", "military aircraft",
    "helicopter", "attack helicopter", "gunship", "chopper",
    "reconnaissance aircraft", "recon plane", "spy plane",
    "drone", "uav", "unmanned aerial vehicle",
    "transport aircraft", "cargo plane", "military transport",
    "air force", "airforce", "aerial warfare",
    "parachute", "paratroop", "airborne",
    "nuclear weapon", "nuclear warhead", "nuclear bomb",
    "chemical weapon", "biological weapon",
    "half track", "halftrack", "military vehicle",
    "combat", "warfare", "battlefield", "weapon", "armament",
}

MILITARY_SYNSETS = {
    "soldier.n.01", "military_officer.n.01", "serviceman.n.01", "troop.n.01",
    "infantry.n.01", "marine.n.01", "paratrooper.n.01", "commando.n.01",
    "tank.n.01", "armored_combat_vehicle.n.01", "military_vehicle.n.01",
    "armored_car.n.01", "half_track.n.01",
    "howitzer.n.01", "cannon.n.01", "artillery.n.01", "mortar.n.02",
    "machine_gun.n.01", "rifle.n.01", "assault_rifle.n.01", "pistol.n.01",
    "grenade.n.01", "missile.n.01", "projectile.n.01", "bomb.n.01",
    "warship.n.01", "battleship.n.01", "destroyer.n.01", "cruiser.n.01",
    "frigate.n.01", "submarine.n.01", "aircraft_carrier.n.01",
    "fighter_aircraft.n.01", "bomber.n.01", "warplane.n.01",
    "attack_aircraft.n.01", "helicopter.n.01", "gunship.n.01",
    "reconnaissance_plane.n.01", "transport_plane.n.01",
    "drone.n.01", "unmanned_aerial_vehicle.n.01",
    "military_uniform.n.01", "camouflage.n.01", "helmet.n.01",
    "body_armor.n.01", "flak_jacket.n.01",
    "bunker.n.01", "fortification.n.01", "pillbox.n.01", "trench.n.01",
    "radar.n.01", "sonar.n.01",
    "parachute.n.01", "airborne.n.01",
    "half_track.n.02", "weapon.n.01", "armament.n.01",
}


class ImageClassifier:
    def __init__(self, threshold=0.05, top_k=20):
        self.threshold = threshold
        self.top_k = top_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.labels = None
        self.military_class_ids = set()
        self.model_loaded = False
        self._load_model()

    def _load_model(self):
        try:
            logger.info("[VIT] Loading ResNet50 with ImageNet-1K weights...")
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.model.eval()
            self.model.to(self.device)
            self.model_loaded = True
            logger.info("[VIT] ResNet50 loaded on %s", self.device)

            self._load_labels()
            self._build_military_ids()
        except Exception as exc:
            logger.exception("[VIT] Failed to load model: %s", exc)
            self.model_loaded = False

    def _load_labels(self):
        labels_path = "imagenet_labels.json"
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                self.labels = json.load(f)
            logger.info("[VIT] Loaded %d labels from %s", len(self.labels), labels_path)
            return

        try:
            logger.info("[VIT] Downloading ImageNet labels from %s ...", IMAGENET_LABELS_URL)
            with urllib.request.urlopen(IMAGENET_LABELS_URL, timeout=15) as resp:
                self.labels = json.loads(resp.read().decode())
            with open(labels_path, "w") as f:
                json.dump(self.labels, f)
            logger.info("[VIT] Downloaded & cached %d labels", len(self.labels))
        except Exception as exc:
            logger.warning("[VIT] Could not download labels: %s", exc)
            self.labels = [f"class_{i}" for i in range(1000)]

    def _build_military_ids(self):
        count = 0
        for idx, label in enumerate(self.labels):
            label_lower = label.lower()
            if any(kw in label_lower for kw in MILITARY_KEYWORDS):
                self.military_class_ids.add(idx)
                count += 1
                continue
            for synset_id in MILITARY_SYNSETS:
                synset_label = synset_id.replace("_", " ").replace(".n.01", "").replace(".n.02", "")
                if synset_label in label_lower or label_lower in synset_label:
                    self.military_class_ids.add(idx)
                    count += 1
                    break
        logger.info("[VIT] Built %d military class IDs (matched %d labels)", len(self.military_class_ids), count)

    def predict(self, img_bgr):
        if not self.model_loaded or self.model is None:
            return self._empty_result()

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:self.top_k]
        military_score = 0.0
        max_military_prob = 0.0
        is_military = False
        top_predictions = []

        for i, idx in enumerate(top_indices):
            p = float(probs[idx])
            label = self.labels[idx] if self.labels and idx < len(self.labels) else f"class_{idx}"
            is_mil_class = idx in self.military_class_ids

            if is_mil_class:
                military_score += p
                if p >= self.threshold:
                    max_military_prob = max(max_military_prob, p)
                    if not is_military:
                        is_military = True

            top_predictions.append({
                "class": label,
                "probability": round(p, 4),
                "is_military": is_mil_class,
            })

        # Also compute military_score from ALL classes, not just top-k
        full_military_score = 0.0
        for idx in self.military_class_ids:
            if idx < len(probs):
                full_military_score += float(probs[idx])

        return {
            "top_predictions": top_predictions,
            "is_military": is_military,
            "confidence": round(max_military_prob, 4),
            "military_score": round(full_military_score, 4),
        }

    def _empty_result(self):
        return {
            "top_predictions": [{"class": "model_not_loaded", "probability": 0.0, "is_military": False}],
            "is_military": False,
            "confidence": 0.0,
            "military_score": 0.0,
        }
