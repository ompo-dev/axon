use std::{
    fmt,
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
};

use crate::{AxonTask, DeltaForge, DerivedArtifact, FoldSpec, ForgeError};

const SCHEMA: &str = "axon-uic-artifact-v1";

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
            Self::Forge(error) => write!(formatter, "cannot derive artifact: {error:?}"),
        }
    }
}

impl std::error::Error for ArtifactStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Forge(error) => Some(error),
            Self::InvalidRecord(_) | Self::MismatchedCapability { .. } => None,
        }
    }
}

/// Filesystem-backed artifact registry. Stored data is declarative metadata; each program is
/// reconstructed from the verified capability grammar on loading.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactStore {
    root: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InstalledArtifact {
    status: ArtifactStatus,
    path: PathBuf,
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
        let path = self.root.join(format!("{}.artifact", task.name()));
        if path.exists() {
            let stored = read_record(&path)?;
            if stored != task.goal() {
                return Err(ArtifactStoreError::MismatchedCapability {
                    path,
                    stored,
                    requested: task.goal(),
                });
            }
            return derive(stored, ArtifactStatus::Reused, path);
        }

        fs::create_dir_all(&self.root).map_err(|source| ArtifactStoreError::Io {
            operation: "create artifact directory",
            path: self.root.clone(),
            source,
        })?;
        write_record(&path, task)?;
        derive(task.goal(), ArtifactStatus::Created, path)
    }
}

impl InstalledArtifact {
    pub const fn status(&self) -> ArtifactStatus {
        self.status
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub const fn artifact(&self) -> DerivedArtifact {
        self.artifact
    }
}

fn derive(
    capability: FoldSpec,
    status: ArtifactStatus,
    path: PathBuf,
) -> Result<InstalledArtifact, ArtifactStoreError> {
    let artifact =
        DeltaForge::synthesize_capability(capability).map_err(ArtifactStoreError::Forge)?;
    Ok(InstalledArtifact {
        status,
        path,
        artifact,
    })
}

fn read_record(path: &Path) -> Result<FoldSpec, ArtifactStoreError> {
    let contents = fs::read_to_string(path).map_err(|source| ArtifactStoreError::Io {
        operation: "read",
        path: path.to_path_buf(),
        source,
    })?;
    let mut lines = contents.lines();
    let schema = lines.next();
    let capability = lines
        .next()
        .and_then(|line| line.strip_prefix("capability="));
    if schema != Some(SCHEMA) || lines.next().is_some() {
        return Err(ArtifactStoreError::InvalidRecord(path.to_path_buf()));
    }
    capability
        .and_then(FoldSpec::from_name)
        .ok_or_else(|| ArtifactStoreError::InvalidRecord(path.to_path_buf()))
}

fn write_record(path: &Path, task: &AxonTask) -> Result<(), ArtifactStoreError> {
    let temporary = path.with_extension("tmp");
    let record = format!("{SCHEMA}\ncapability={}\n", task.goal().as_str());
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
