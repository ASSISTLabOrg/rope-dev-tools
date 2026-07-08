import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ROPE_REGISTRY_DEFAULT = REPO_ROOT.parent / "rope-registry"


@pytest.fixture(scope="session", autouse=True)
def _rope_registry_path_default():
    if "ROPE_REGISTRY_PATH" not in os.environ and ROPE_REGISTRY_DEFAULT.is_dir():
        os.environ["ROPE_REGISTRY_PATH"] = str(ROPE_REGISTRY_DEFAULT)
    yield


@pytest.fixture(scope="session")
def registry_root():
    from rope_dev_tools.registry.vendor import RegistryVendor

    return RegistryVendor().resolve()


@pytest.fixture(scope="session")
def schema_store(registry_root):
    from rope_dev_tools.registry.schema_store import RegistrySchemaStore

    return RegistrySchemaStore(registry_root)


@pytest.fixture(scope="session")
def validator(schema_store):
    from rope_dev_tools.registry.validate import ManifestValidator

    return ManifestValidator(schema_store)
