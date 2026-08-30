//! Identificadores estáveis e serializáveis por valor para o núcleo V6.

macro_rules! identifier {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
        pub struct $name(pub u64);
    };
}

identifier!(FactorId);
identifier!(ClaimId);
identifier!(RevisionId);
identifier!(ProgramId);
identifier!(PatchId);
