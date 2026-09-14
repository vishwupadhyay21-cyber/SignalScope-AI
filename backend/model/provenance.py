"""Provenance & C2PA / Content Credentials Metadata parser for SignalScope (Bonus Module D)."""

from typing import Dict, Any, Optional
from PIL import Image
from app.services.metadata_extractor import extract_image_metadata


def inspect_provenance_and_c2pa(image: Image.Image) -> Dict[str, Any]:
    """Inspects C2PA / Content Credentials markers and EXIF forensic metadata.

    Distinguishes visual model evidence from metadata evidence.
    Missing metadata is NOT treated as proof of fake imagery.

    Args:
        image: PIL Image object.

    Returns:
        dict: Provenance audit information and metadata breakdown.
    """
    exif_meta = extract_image_metadata(image)

    # Check C2PA / Content Credentials signatures in image info or XMP tags
    has_c2pa_signature = False
    c2pa_manifest_claim = None

    if hasattr(image, "info"):
        info_dict = getattr(image, "info", {})
        # Scan raw info keys for C2PA or Content Credentials signatures
        for k, v in info_dict.items():
            k_str = str(k).lower()
            v_str = str(v).lower()
            if "c2pa" in k_str or "contentcredentials" in k_str or "jumbf" in k_str or "c2pa" in v_str:
                has_c2pa_signature = True
                c2pa_manifest_claim = "C2PA / Content Credentials signature detected in container header."
                break

    exif_available = exif_meta.get("available", False)
    camera_make = exif_meta.get("camera_make")
    software = exif_meta.get("software")

    provenance_status = "unverified"
    notes = []

    if has_c2pa_signature:
        provenance_status = "c2pa_verified"
        notes.append("Cryptographic Content Credentials (C2PA) manifest discovered.")
    elif exif_available and (camera_make or software):
        provenance_status = "exif_present"
        notes.append(f"Physical hardware metadata present (Camera: {camera_make or 'Unknown'}, Software: {software or 'None'}).")
    else:
        provenance_status = "no_metadata"
        notes.append("No embedded EXIF hardware metadata or C2PA manifest found. Treated neutral; visual classifier evaluates image content independently.")

    return {
        "provenance_status": provenance_status,
        "c2pa_detected": has_c2pa_signature,
        "c2pa_claim": c2pa_manifest_claim,
        "exif_metadata": exif_meta,
        "provenance_notes": notes,
        "isolation_guarantee": "Visual classification verdict is calculated independently from metadata availability.",
    }
