use std::{
    fmt,
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
};

use crate::{
    AxonTask, CertificateStatus, DeltaCertificate, DeltaForge, DerivedArtifact, FoldSpec,
    ForgeError, PhysicalRealization, RuntimeGuards, SemanticArtifact, SemanticArtifactError,
    SemanticArtifactHash,
};

const SCHEMA: &str = "axon-uic-semantic-artifact-v2";
const CACHE_FILE_VERSION: u16 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArtifactStatus {
    Created,
    Reused,
}

impl ArtifactStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Reused => "reused",
        }
    }
}

#[derive(Debug)]
pub enum ArtifactStoreError {
    Io {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    InvalidRecord(PathBuf),
    MismatchedCapability {
        path: PathBuf,
        stored: FoldSpec,
        requested: FoldSpec,
    },
    Semantic(SemanticArtifactError),
    Forge(ForgeError),
}

impl fmt::Display for ArtifactStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                operation, path, ..
            } => write!(
                formatter,
                "cannot {operation} artifact `{}`",
                path.display()
            ),
            Self::InvalidRecord(path) => {
                write!(formatter, "invalid artifact record `{}`", path.display())
            }
            Self::MismatchedCapability {
                path,
                stored,
                requested,
            } => write!(
                formatter,
                "artifact `{}` has capability {}; task requests {}",
                path.display(),
                stored.as_str(),
                requested.as_str()
            ),
            Self::Semantic(error) => write!(formatter, "invalid semantic artifact: {error}"),
            Self::Forge(error) => write!(formatter, "cannot realize artifact: {error}"),
        }
    }
}

impl std::error::Error for ArtifactStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Semantic(error) => Some(error),
            Self::Forge(error) => Some(error),
            Self::InvalidRecord(_) | Self::MismatchedCapability { .. } => None,
        }
    }
}

/// Filesystem-backed semantic registry. It persists only immutable semantics and seal.
/// Physical realization is rebuilt for current hardware.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactStore {
    root: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InstalledArtifact {
    status: ArtifactStatus,
    certificate_status: CertificateStatus,
    path: PathBuf,
    semantic: SemanticArtifact,
    physical: PhysicalRealization,
    artifact: DerivedArtifact,
}

impl ArtifactStore {
    pub fn open(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn install(&self, task: &AxonTask) -> Result<InstalledArtifact, ArtifactStoreError> {
        let path = self.artifact_path(task);
        if path.exists() {
            let semantic = read_record(&path)?;
            if semantic.capability() != task.goal() {
                return Err(ArtifactStoreError::MismatchedCapability {
                    path,
                    stored: semantic.capability(),
                    requested: task.goal(),
                });
            }
            return realize(
                semantic,
                ArtifactStatus::Reused,
                CertificateStatus::Cached,
                path,
            );
        }

        fs::create_dir_all(&self.root).map_err(|source| ArtifactStoreError::Io {
            operation: "create artifact directory",
            path: self.root.clone(),
            source,
        })?;
        let semantic =
            SemanticArtifact::synthesize(task.goal()).map_err(ArtifactStoreError::Semantic)?;
        write_record(&path, semantic)?;
        realize(
            semantic,
            ArtifactStatus::Created,
            CertificateStatus::Verified,
            path,
        )
    }

    /// Separating the cache filename version lets a derived legacy cache age out safely instead
    /// of making a newly installed semantic schema reject every future solve.
    #[must_use]
    fn artifact_path(&self, task: &AxonTask) -> PathBuf {
        self.root.join(format!(
            "{}.semantic-v{CACHE_FILE_VERSION}.artifact",
            task.name()
        ))
    }
}

impl InstalledArtifact {
    pub const fn status(&self) -> ArtifactStatus {
        self.status
    }

    pub const fn certificate_status(&self) -> CertificateStatus {
        self.certificate_status
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub const fn semantic(&self) -> SemanticArtifact {
        self.semantic
    }

    pub const fn physical(&self) -> PhysicalRealization {
        self.physical
    }

    pub const fn artifact(&self) -> DerivedArtifact {
        self.artifact
    }
}

fn realize(
    semantic: SemanticArtifact,
    status: ArtifactStatus,
    certificate_status: CertificateStatus,
    path: PathBuf,
) -> Result<InstalledArtifact, ArtifactStoreError> {
    let artifact = DeltaForge::synthesize_capability(semantic.capability())
        .map_err(ArtifactStoreError::Forge)?;
    Ok(InstalledArtifact {
        status,
        certificate_status,
        path,
        semantic,
        physical: PhysicalRealization::interpreter(semantic.hash()),
        artifact,
    })
}

fn read_record(path: &Path) -> Result<SemanticArtifact, ArtifactStoreError> {
    let contents = fs::read_to_string(path).map_err(|source| ArtifactStoreError::Io {
        operation: "read",
        path: path.to_path_buf(),
        source,
    })?;
    let mut lines = contents.lines();
    let invalid = || ArtifactStoreError::InvalidRecord(path.to_path_buf());
    if lines.next() != Some(SCHEMA) {
        return Err(invalid());
    }
    let capability = field(lines.next(), "capability=")
        .and_then(FoldSpec::from_name)
        .ok_or_else(invalid)?;
    let kernel_version = field(lines.next(), "kernel_semantics_version=")
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(invalid)?;
    let semantic_version = field(lines.next(), "semantic_artifact_version=")
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(invalid)?;
    let certificate = field(lines.next(), "certificate=")
        .and_then(|value| DeltaCertificate::from_identifier(capability, value))
        .ok_or_else(invalid)?;
    let guards = field(lines.next(), "guards=")
        .and_then(|value| RuntimeGuards::from_identifier(capability, value))
        .ok_or_else(invalid)?;
    let hash = field(lines.next(), "semantic_hash=")
        .and_then(SemanticArtifactHash::from_hex)
        .ok_or_else(invalid)?;
    if lines.next().is_some() {
        return Err(invalid());
    }
    SemanticArtifact::from_record(
        capability,
        certificate,
        guards,
        kernel_version,
        semantic_version,
        hash,
    )
    .map_err(ArtifactStoreError::Semantic)
}

fn field<'a>(line: Option<&'a str>, prefix: &str) -> Option<&'a str> {
    line.and_then(|line| line.strip_prefix(prefix))
}

fn write_record(path: &Path, semantic: SemanticArtifact) -> Result<(), ArtifactStoreError> {
    let temporary = path.with_extension("tmp");
    let record = format!(
        "{SCHEMA}\ncapability={}\nkernel_semantics_version={}\nsemantic_artifact_version={}\ncertificate={}\nguards={}\nsemantic_hash={}\n",
        semantic.capability().as_str(),
        semantic.kernel_version(),
        semantic.semantic_version(),
        semantic.certificate().identifier(),
        semantic.guards().identifier(),
        semantic.hash(),
    );
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|source| ArtifactStoreError::Io {
            operation: "create temporary",
            path: temporary.clone(),
            source,
        })?;
    file.write_all(record.as_bytes())
        .and_then(|()| file.sync_all())
        .map_err(|source| ArtifactStoreError::Io {
            operation: "write temporary",
            path: temporary.clone(),
            source,
        })?;
    fs::rename(&temporary, path).map_err(|source| ArtifactStoreError::Io {
        operation: "publish",
        path: path.to_path_buf(),
        source,
    })
}
