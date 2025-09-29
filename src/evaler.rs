//! This module evaluates parsed `Expression`s and compiled `Instruction`s.
//!
//! Everything can be evaluated using the `.eval()` method, but compiled
//! `Instruction`s also have the option of using the `eval_compiled!()` macro
//! which is much faster for common cases.



use crate as fasteval;

use crate::error::Error;
use crate::slab::Slab;
use crate::evalns::{EvalNamespace, TargetContext};
use crate::parser::{Expression,
                    Value::{self, EConstant, EUnaryOp, EStdFunc, EPrintFunc},
                    UnaryOp::{self, EPos, ENeg, ENot, EParentheses},
                    BinaryOp::{self, EAdd, ESub, EMul, EDiv, EMod, EExp, ELT, ELTE, EEQ, ENE, EGTE, EGT, EOR, EAND},
                    StdFunc::{self, EVar, EFunc, EFuncInt, EFuncCeil, EFuncFloor, EFuncAbs, EFuncSign, EFuncLog, EFuncRound, EFuncMin, EFuncMax, EFuncE, EFuncPi, EFuncSin, EFuncCos, EFuncTan, EFuncASin, EFuncACos, EFuncATan, EFuncSinH, EFuncCosH, EFuncTanH, EFuncASinH, EFuncACosH, EFuncATanH},
                    PrintFunc,
                    ExpressionOrString::{EExpr, EStr},
                    remove_no_panic};
#[cfg(feature="unsafe-vars")]
use crate::parser::StdFunc::EUnsafeVar;
use crate::compiler::{log, IC, Instruction::{self, IConst, INeg, INot, IInv, IAdd, IMul, IMod, IExp, ILT, ILTE, IEQ, INE, IGTE, IGT, IOR, IAND, IVar, IFunc, IFuncInt, IFuncCeil, IFuncFloor, IFuncAbs, IFuncSign, IFuncLog, IFuncRound, IFuncMin, IFuncMax, IFuncSin, IFuncCos, IFuncTan, IFuncASin, IFuncACos, IFuncATan, IFuncSinH, IFuncCosH, IFuncTanH, IFuncASinH, IFuncACosH, IFuncATanH, IPrintFunc}};
#[cfg(feature="unsafe-vars")]
use crate::compiler::Instruction::IUnsafeVar;

use std::collections::BTreeSet;
use std::f64::consts;
use std::fmt;
use crate::Instruction::{IFuncIf, ITargetVar};
use crate::parser::StdFunc::{EFuncIf, ETargetVar};

/// The same as `evaler.eval(&slab, &mut ns)`, but more efficient for common cases.
///
/// This macro is exactly the same as [`eval_compiled_ref!()`](macro.eval_compiled_ref.html)
/// but is more efficient if you have ownership of the evaler.
///
/// Only use this for compiled expressions.  (If you use it for interpreted
/// expressions, it will work but will always be slower than calling `eval()` directly.)
///
/// This macro is able to eliminate function calls for constants and Unsafe Variables.
/// Since evaluation is a performance-critical operation, saving some function
/// calls actually makes a huge performance difference.
///
#[macro_export]
macro_rules! eval_compiled {
    ($evaler:ident, $slab_ref:expr, $ns_mut:expr, $tc_ref:expr) => {
        if let fasteval::IConst(c) = $evaler {
            c
        } else {
            #[cfg(feature="unsafe-vars")]
            {
                if let fasteval::IUnsafeVar{ptr, ..} = $evaler {
                    unsafe { *ptr }
                } else {
                    $evaler.eval($slab_ref, $ns_mut, $tc_ref)?
                }
            }

            #[cfg(not(feature="unsafe-vars"))]
            $evaler.eval($slab_ref, $ns_mut, $tc_ref)?
        }
    };
    ($evaler:expr, $slab_ref:expr, $ns_mut:expr, $tc_ref:expr) => {
        {
            let evaler = $evaler;
            eval_compiled!(evaler, $slab_ref, $ns_mut, $tc_ref)
        }
    };
}

/// The same as `evaler_ref.eval(&slab, &mut ns)`, but more efficient for common cases.
///
/// This macro is exactly the same as [`eval_compiled!()`](macro.eval_compiled.html) but
/// is useful when you hold a reference to the evaler, rather than having ownership of it.
///
/// Only use this for compiled expressions.  (If you use it for interpreted
/// expressions, it will work but will always be slower than calling `eval()` directly.)
///
/// This macro is able to eliminate function calls for constants and Unsafe Variables.
/// Since evaluation is a performance-critical operation, saving some function
/// calls actually makes a huge performance difference.
///
#[macro_export]
macro_rules! eval_compiled_ref {
    ($evaler:ident, $slab_ref:expr, $ns_mut:expr, $tc_ref:expr) => {
        if let fasteval::IConst(c) = $evaler {
            *c
        } else {
            #[cfg(feature="unsafe-vars")]
            {
                if let fasteval::IUnsafeVar{ptr, ..} = $evaler {
                    unsafe { **ptr }
                } else {
                    $evaler.eval($slab_ref, $ns_mut, $tc_ref)?
                }
            }

            #[cfg(not(feature="unsafe-vars"))]
            $evaler.eval($slab_ref, $ns_mut, $tc_ref)?
        }
    };
    ($evaler:expr, $slab_ref:expr, $ns_mut:expr, $tc_ref:expr) => {
        {
            let evaler = $evaler;
            eval_compiled_ref!(evaler, $slab_ref, $ns_mut, $tc_ref)
        }
    };
}

macro_rules! eval_ic_ref {
    ($ic:ident, $slab_ref:ident, $ns_mut:expr, $tc_ref:expr) => {
        match $ic {
            IC::C(c) => *c,
            IC::I(i) => {
                let instr_ref = get_instr!($slab_ref.cs,i);

                #[cfg(feature="unsafe-vars")]
                {
                    if let fasteval::IUnsafeVar{ptr, ..} = instr_ref {
                        unsafe { **ptr }
                    } else {
                        instr_ref.eval($slab_ref, $ns_mut, $tc_ref)?
                    }
                }

                #[cfg(not(feature="unsafe-vars"))]
                instr_ref.eval($slab_ref, $ns_mut, $tc_ref)?
            }
        }
    }
}



/// You must `use` this trait so you can call `.eval()`.
pub trait Evaler : fmt::Debug {
    /// Evaluate this `Expression`/`Instruction` and return an `f64`.
    ///
    /// Returns a `fasteval::Error` if there are any problems, such as undefined variables.
    fn eval(&self, slab:&Slab, ns:&mut impl EvalNamespace, tc:&impl TargetContext) -> Result<f64,Error>;

    /// Don't call this directly.  Use `var_names()` instead.
    ///
    /// This exists because of ternary short-circuits; they prevent us from
    /// getting a complete list of vars just by doing eval() with a clever
    /// callback.
    fn _var_names(&self, slab:&Slab, dst:&mut BTreeSet<String>);

    /// Returns a list of variables and custom functions that are used by this `Expression`/`Instruction`.
    fn var_names(&self, slab:&Slab) -> BTreeSet<String> {
        let mut set = BTreeSet::new();
        self._var_names(slab,&mut set);
        set
    }
}

impl Evaler for Expression {
    fn _var_names(&self, slab:&Slab, dst:&mut BTreeSet<String>) {
        self.first._var_names(slab,dst);
        for pair in &self.pairs {
            pair.1._var_names(slab,dst);
        }
    }
    fn eval(&self, slab:&Slab, ns:&mut impl EvalNamespace, tc:&impl TargetContext) -> Result<f64,Error> {
        // Order of operations: 1) ^  2) */  3) +-
        // Exponentiation should be processed right-to-left.  Think of what 2^3^4 should mean:
        //     2^(3^4)=2417851639229258349412352   <--- I choose this one.  https://codeplea.com/exponentiation-associativity-options
        //     (2^3)^4=4096
        // Direction of processing doesn't matter for Addition and Multiplication:
        //     (((3+4)+5)+6)==(3+(4+(5+6))), (((3*4)*5)*6)==(3*(4*(5*6)))
        // ...But Subtraction and Division must be processed left-to-right:
        //     (((6-5)-4)-3)!=(6-(5-(4-3))), (((6/5)/4)/3)!=(6/(5/(4/3)))


        // // ---- Go code, for comparison ----
        // // vals,ops:=make([]float64, len(e)/2+1),make([]BinaryOp, len(e)/2)
        // // for i:=0; i<len(e); i+=2 {
        // //     vals[i/2]=ns.EvalBubble(e[i].(evaler))
        // //     if i<len(e)-1 { ops[i/2]=e[i+1].(BinaryOp) }
        // // }

        // if self.0.len()%2!=1 { return Err(KErr::new("Expression len should always be odd")) }
        // let mut vals : Vec<f64>      = Vec::with_capacity(self.0.len()/2+1);
        // let mut ops  : Vec<BinaryOp> = Vec::with_capacity(self.0.len()/2  );
        // for (i,tok) in self.0.iter().enumerate() {
        //     match tok {
        //         EValue(val) => {
        //             if i%2==1 { return Err(KErr::new("Found value at odd index")) }
        //             match ns.eval_bubble(val) {
        //                 Ok(f) => vals.push(f),
        //                 Err(e) => return Err(e.pre(&format!("eval_bubble({:?})",val))),
        //             }
        //         }
        //         EBinaryOp(bop) => {
        //             if i%2==0 { return Err(KErr::new("Found binaryop at even index")) }
        //             ops.push(*bop);
        //         }
        //     }
        // }

        // Code for new Expression data structure:
        let mut vals = Vec::<f64>::with_capacity(self.pairs.len()+1);
        let mut ops  = Vec::<BinaryOp>::with_capacity(self.pairs.len());
        vals.push(self.first.eval(slab,ns, tc)?);
        for pair in self.pairs.iter() {
            ops.push(pair.0);
            vals.push(pair.1.eval(slab,ns, tc)?);
        }


        // ---- Go code, for comparison ----
        // evalOp:=func(i int) {
        //     result:=ops[i]._Eval(vals[i], vals[i+1])
        //     vals=append(append(vals[:i], result), vals[i+2:]...)
        //     ops=append(ops[:i], ops[i+1:]...)
        // }
        // rtol:=func(s BinaryOp) { for i:=len(ops)-1; i>=0; i-- { if ops[i]==s { evalOp(i) } } }
        // ltor:=func(s BinaryOp) {
        //     loop:
        //     for i:=0; i<len(ops); i++ { if ops[i]==s { evalOp(i); goto loop } }  // Need to restart processing when modifying from the left.
        // }

        #[inline(always)]
        fn rtol(vals:&mut Vec<f64>, ops:&mut Vec<BinaryOp>, search:BinaryOp) {
            for i in (0..ops.len()).rev() {
                let op = match ops.get(i) {
                    Some(op) => *op,
                    None => EOR,  // unreachable
                };
                if op==search {
                    let res = op.binaryop_eval(vals.get(i), vals.get(i+1));
                    match vals.get_mut(i) {
                        Some(val_ref) => *val_ref=res,
                        None => (),  // unreachable
                    };
                    remove_no_panic(vals, i+1);
                    remove_no_panic(ops, i);
                }
            }
        }
        #[inline(always)]
        fn ltor(vals:&mut Vec<f64>, ops:&mut Vec<BinaryOp>, search:BinaryOp) {
            let mut i = 0;
            loop {
                match ops.get(i) {
                    None => break,
                    Some(op) => {
                        if *op==search {
                            let res = op.binaryop_eval(vals.get(i), vals.get(i+1));
                            match vals.get_mut(i) {
                                Some(val_ref) => *val_ref=res,
                                None => (),  // unreachable
                            };
                            remove_no_panic(vals, i+1);
                            remove_no_panic(ops, i);
                        } else {
                            i=i+1;
                        }
                    }
                }
            }
        }
        #[inline(always)]
        fn ltor_multi(vals:&mut Vec<f64>, ops:&mut Vec<BinaryOp>, search:&[BinaryOp]) {
            let mut i = 0;
            loop {
                match ops.get(i) {
                    None => break,
                    Some(op) => {
                        if search.contains(op) {
                            let res = op.binaryop_eval(vals.get(i), vals.get(i+1));
                            match vals.get_mut(i) {
                                Some(val_ref) => *val_ref=res,
                                None => (),  // unreachable
                            };
                            remove_no_panic(vals, i+1);
                            remove_no_panic(ops, i);
                        } else {
                            i=i+1;
                        }
                    }
                }
            }
        }

        // Keep the order of these statements in-sync with parser.rs BinaryOp priority values:
        rtol(&mut vals, &mut ops, EExp);  // https://codeplea.com/exponentiation-associativity-options
        ltor(&mut vals, &mut ops, EMod);
        ltor(&mut vals, &mut ops, EDiv);
        rtol(&mut vals, &mut ops, EMul);
        ltor(&mut vals, &mut ops, ESub);
        rtol(&mut vals, &mut ops, EAdd);
        ltor_multi(&mut vals, &mut ops, &[ELT, EGT, ELTE, EGTE, EEQ, ENE]);  // TODO: Implement Python-style a<b<c ternary comparison... might as well generalize to N comparisons.
        ltor(&mut vals, &mut ops, EAND);
        ltor(&mut vals, &mut ops, EOR);

        if !ops.is_empty() { return Err(Error::Unreachable); }
        if vals.len()!=1 { return Err(Error::Unreachable); }
        match vals.first() {
            Some(val) => Ok(*val),
            None => Err(Error::Unreachable),
        }
    }
}

impl Evaler for Value {
    fn _var_names(&self, slab:&Slab, dst:&mut BTreeSet<String>) {
        match self {
            EConstant(_) => (),
            EUnaryOp(u) => u._var_names(slab,dst),
            EStdFunc(f) => f._var_names(slab,dst),
            EPrintFunc(f) => f._var_names(slab,dst),
        };
    }
    fn eval(&self, slab:&Slab, ns:&mut impl EvalNamespace, tc:&impl TargetContext) -> Result<f64,Error> {
        match self {
            EConstant(c) => Ok(*c),
            EUnaryOp(u) => u.eval(slab,ns, tc),
            EStdFunc(f) => f.eval(slab,ns, tc),
            EPrintFunc(f) => f.eval(slab,ns, tc),
        }
    }
}

impl Evaler for UnaryOp {
    fn _var_names(&self, slab:&Slab, dst:&mut BTreeSet<String>) {
        match self {
            EPos(val_i) | ENeg(val_i) | ENot(val_i) => get_val!(slab.ps,val_i)._var_names(slab,dst),
            EParentheses(expr_i) => get_expr!(slab.ps,expr_i)._var_names(slab,dst),
        }
    }
    fn eval(&self, slab:&Slab, ns:&mut impl EvalNamespace, tc:&impl TargetContext) -> Result<f64,Error> {
        match self {
            EPos(val_i) => get_val!(slab.ps,val_i).eval(slab,ns, tc),
            ENeg(val_i) => Ok(-get_val!(slab.ps,val_i).eval(slab,ns, tc)?),
            ENot(val_i) => Ok(bool_to_f64!(f64_eq!(get_val!(slab.ps,val_i).eval(slab,ns, tc)?,0.0))),
            EParentheses(expr_i) => get_expr!(slab.ps,expr_i).eval(slab,ns, tc),
        }
    }
}

impl BinaryOp {
    // Non-standard eval interface (not generalized yet):
    fn binaryop_eval(self, left_opt:Option<&f64>, right_opt:Option<&f64>) -> f64 {  // Passing 'self' by value is more efficient than pass-by-reference.
        let left = match left_opt {
            Some(l) => *l,
            None => return std::f64::NAN,
        };
        let right = match right_opt {
            Some(r) => *r,
            None => return std::f64::NAN,
        };
        match self {
            EAdd => left+right,  // Floats don't overflow.
            ESub => left-right,
            EMul => left*right,
            EDiv => left/right,
            EMod => left%right, //left - (left/right).trunc()*right
            EExp => left.powf(right),
            ELT => bool_to_f64!(left<right),
            ELTE => bool_to_f64!(left<=right),
            EEQ => bool_to_f64!(f64_eq!(left,right)),
            ENE => bool_to_f64!(f64_ne!(left,right)),
            EGTE => bool_to_f64!(left>=right),
            EGT => bool_to_f64!(left>right),
            EOR => if f64_ne!(left,0.0) { left }
                   else { right },
            EAND => if f64_eq!(left,0.0) { left }
                    else { right },
        }
    }
}

macro_rules! eval_var {
    ($ns:ident, $name:ident, $args:expr, $keybuf:expr) => {
        match $ns.lookup($name,$args,$keybuf) {
            Some(f) => Ok(f),
            None => Err(Error::Undefined($name.to_string())),
        }
    };
}

impl Evaler for StdFunc {
    fn _var_names(&self, slab:&Slab, dst:&mut BTreeSet<String>) {
        match self {
            #[cfg(feature="unsafe-vars")]
            EUnsafeVar{name, ..} => { dst.insert(name.clone()); }

            EVar(s) => { dst.insert(s.clone()); }
            ETargetVar(s) => { dst.insert(s.clone()); }
            EFunc{name, args} => {
                dst.insert(name.clone());
                for arg in args {
                    get_expr!(slab.ps,arg)._var_names(slab,dst);
                }
            }

            EFuncInt(xi) | EFuncCeil(xi) | EFuncFloor(xi) | EFuncAbs(xi) | EFuncSign(xi) | EFuncSin(xi) | EFuncCos(xi) | EFuncTan(xi) | EFuncASin(xi) | EFuncACos(xi) | EFuncATan(xi) | EFuncSinH(xi) | EFuncCosH(xi) | EFuncTanH(xi) | EFuncASinH(xi) | EFuncACosH(xi) | EFuncATanH(xi) => get_expr!(slab.ps,xi)._var_names(slab,dst),

            EFuncE | EFuncPi => (),

            EFuncLog{base:opt,expr} | EFuncRound{modulus:opt,expr} => {
                match opt {
                    Some(xi) => get_expr!(slab.ps,xi)._var_names(slab,dst),
                    None => (),
                }
                get_expr!(slab.ps,expr)._var_names(slab,dst);
            }
            EFuncMin{first,rest} | EFuncMax{first,rest} => {
                get_expr!(slab.ps,first)._var_names(slab,dst);
                for xi in rest {
                    get_expr!(slab.ps,xi)._var_names(slab,dst);
                }
            }
            EFuncIf {cond:ci, then:ti, els:ei} => {
                get_expr!(slab.ps,ci )._var_names(slab,dst);
                get_expr!(slab.ps,ti)._var_names(slab,dst);
                get_expr!(slab.ps,ei)._var_names(slab,dst);
            }
        };
    }
    fn eval(&self, slab:&Slab, ns:&mut impl EvalNamespace, tc:&impl TargetContext) -> Result<f64,Error> {
        match self {
            // These match arms are ordered in a way that I feel should deliver good performance.
            // (I don't think this ordering actually affects the generated code, though.)

            #[cfg(feature="unsafe-vars")]
            EUnsafeVar{ptr, ..} => unsafe { Ok(**ptr) },

            EVar(name) => eval_var!(ns, name, Vec::new(), unsafe{ &mut *(std::ptr::addr_of!(slab.ps.char_buf) as *mut _) }),
            ETargetVar(name) => {
                match tc.lookup(name) {
                    Some(f) => Ok(f),
                    None => Err(Error::Undefined(name.to_string())),
                }
            }
            EFunc{name, args:xis} => {
                let mut args = Vec::with_capacity(xis.len());
                for xi in xis {
                    args.push(get_expr!(slab.ps,xi).eval(slab,ns,tc)?)
                }
                eval_var!(ns, name, args, unsafe{ &mut *(std::ptr::addr_of!(slab.ps.char_buf) as *mut _) })
            }

            EFuncIf {cond:ci, then:ti, els:ei} => {
                if f64_ne!(get_expr!(slab.ps,ci).eval(slab,ns, tc)?,0.0) {
                    get_expr!(slab.ps,ti).eval(slab,ns, tc)
                } else {
                    get_expr!(slab.ps,ei).eval(slab,ns, tc)
                }
            }

            EFuncLog{base:base_opt, expr:expr_i} => {
                let base = match base_opt {
                    Some(b_expr_i) => get_expr!(slab.ps,b_expr_i).eval(slab,ns, tc)?,
                    None => 10.0,
                };
                let n = get_expr!(slab.ps,expr_i).eval(slab,ns, tc)?;
                Ok(log(base,n))
            }

            EFuncSin(expr_i) => Ok(get_expr!(slab.ps,expr_i).eval(slab,ns, tc)?.sin()),
            EFuncCos(expr_i) => Ok(get_expr!(slab.ps,expr_i).eval(slab,ns, tc)?.cos()),
            EFuncTan(expr_i) => Ok(get_expr!(slab.ps,expr_i).eval(slab,ns, tc)?.tan()),
            EFuncASin(expr_i) => Ok(get_expr!(slab.ps,expr_i).eval(slab,ns, tc)?.asin()),
            EFuncACos(expr_i) => Ok(get_expr!(slab.ps,expr_i).eval(slab,ns, tc)?.acos()),
            EFuncATan(expr_i) => Ok(get_expr!(slab.ps,expr_i).eval(slab,ns, tc)?.atan()),
            EFuncSinH(expr_i) => Ok(get_expr!(slab.ps,expr_i).eval(slab,ns, tc)?.sinh()),
            EFuncCosH(expr_i) => Ok(get_expr!(slab.ps,expr_i).eval(slab,ns, tc)?.cosh()),
            EFuncTanH(expr_i) => Ok(get_expr!(slab.ps,expr_i).eval(slab,ns, tc)?.tanh()),
            EFuncASinH(expr_i) => Ok(get_expr!(slab.ps,expr_i).eval(slab,ns, tc)?.asinh()),
            EFuncACosH(expr_i) => Ok(get_expr!(slab.ps,expr_i).eval(slab,ns, tc)?.acosh()),
            EFuncATanH(expr_i) => Ok(get_expr!(slab.ps,expr_i).eval(slab,ns, tc)?.atanh()),

            EFuncRound{modulus:modulus_opt, expr:expr_i} => {
                let modulus = match modulus_opt {
                    Some(m_expr_i) => get_expr!(slab.ps,m_expr_i).eval(slab,ns, tc)?,
                    None => 1.0,
                };
                Ok((get_expr!(slab.ps,expr_i).eval(slab,ns, tc)?/modulus).round() * modulus)
            }

            EFuncAbs(expr_i) => Ok(get_expr!(slab.ps,expr_i).eval(slab,ns, tc)?.abs()),
            EFuncSign(expr_i) => Ok(get_expr!(slab.ps,expr_i).eval(slab,ns, tc)?.signum()),
            EFuncInt(expr_i) => Ok(get_expr!(slab.ps,expr_i).eval(slab,ns, tc)?.trunc()),
            EFuncCeil(expr_i) => Ok(get_expr!(slab.ps,expr_i).eval(slab,ns, tc)?.ceil()),
            EFuncFloor(expr_i) => Ok(get_expr!(slab.ps,expr_i).eval(slab,ns, tc)?.floor()),
            EFuncMin{first:first_i, rest} => {
                let mut min = get_expr!(slab.ps,first_i).eval(slab,ns, tc)?;
                let mut saw_nan = min.is_nan();
                for x_i in rest.iter() {
                    min = min.min(get_expr!(slab.ps,x_i).eval(slab,ns, tc)?);
                    saw_nan = saw_nan || min.is_nan();
                }
                if saw_nan { Ok(std::f64::NAN)
                } else { Ok(min) }
            }
            EFuncMax{first:first_i, rest} => {
                let mut max = get_expr!(slab.ps,first_i).eval(slab,ns, tc)?;
                let mut saw_nan = max.is_nan();
                for x_i in rest.iter() {
                    max = max.max(get_expr!(slab.ps,x_i).eval(slab,ns, tc)?);
                    saw_nan = saw_nan || max.is_nan();
                }
                if saw_nan { Ok(std::f64::NAN)
                } else { Ok(max) }
            }

            EFuncE => Ok(consts::E),
            EFuncPi => Ok(consts::PI),
        }
    }
}

impl Evaler for PrintFunc {
    fn _var_names(&self, slab:&Slab, dst:&mut BTreeSet<String>) {
        for x_or_s in &self.0 {
            match x_or_s {
                EExpr(xi) => get_expr!(slab.ps,xi)._var_names(slab,dst),
                EStr(_) => (),
            };
        }
    }
    fn eval(&self, slab:&Slab, ns:&mut impl EvalNamespace, tc:&impl TargetContext) -> Result<f64,Error> {
        let mut val = 0f64;

        fn process_str(s:&str) -> String {
            s.replace("\\n","\n").replace("\\t","\t")
        }

        if let Some(EStr(fmtstr)) = self.0.first() {
            if fmtstr.contains('%') {
                // printf mode:

                //let fmtstr = process_str(fmtstr);

                return Err(Error::WrongArgs("printf formatting is not yet implemented".to_string()));  // TODO: Make a pure-rust sprintf libarary.

                //return Ok(val);
            }
        }

        // Normal Mode:
        let mut out = String::with_capacity(16);
        for (i,a) in self.0.iter().enumerate() {
            if i>0 { out.push(' '); }
            match a {
                EExpr(e_i) => {
                    val = get_expr!(slab.ps,e_i).eval(slab,ns, tc)?;
                    out.push_str(&val.to_string());
                }
                EStr(s) => out.push_str(&process_str(s))
            }
        }
        eprintln!("{}", out);

        Ok(val)
    }
}

impl Evaler for Instruction {
    fn _var_names(&self, slab:&Slab, dst:&mut BTreeSet<String>) {
        match self {
            #[cfg(feature="unsafe-vars")]
            IUnsafeVar{name, ..} => { dst.insert(name.clone()); }

            IVar(s) => { dst.insert(s.clone()); }
            ITargetVar(s) => { dst.insert(s.clone()); }
            IFunc{name, args} => {
                dst.insert(name.clone());
                for ic in args {
                    let iconst : Instruction;
                    ic_to_instr!(slab.cs,iconst,ic)._var_names(slab,dst);
                }
            }

            IConst(_) => (),

            INeg(ii) | INot(ii) | IInv(ii) | IFuncInt(ii) | IFuncCeil(ii) | IFuncFloor(ii) | IFuncAbs(ii) | IFuncSign(ii) | IFuncSin(ii) | IFuncCos(ii) | IFuncTan(ii) | IFuncASin(ii) | IFuncACos(ii) | IFuncATan(ii) | IFuncSinH(ii) | IFuncCosH(ii) | IFuncTanH(ii) | IFuncASinH(ii) | IFuncACosH(ii) | IFuncATanH(ii) => get_instr!(slab.cs,ii)._var_names(slab,dst),

            ILT(lic,ric) | ILTE(lic,ric) | IEQ(lic,ric) | INE(lic,ric) | IGTE(lic,ric) | IGT(lic,ric) | IMod{dividend:lic, divisor:ric} | IExp{base:lic, power:ric} | IFuncLog{base:lic, of:ric} | IFuncRound{modulus:lic, of:ric} => {
                let mut iconst : Instruction;
                ic_to_instr!(slab.cs,iconst,lic)._var_names(slab,dst);
                ic_to_instr!(slab.cs,iconst,ric)._var_names(slab,dst);
            }

            IAdd(li,ric) | IMul(li,ric) | IOR(li,ric) | IAND(li,ric) | IFuncMin(li,ric) | IFuncMax(li,ric) => {
                get_instr!(slab.cs,li)._var_names(slab,dst);
                let iconst : Instruction;
                ic_to_instr!(slab.cs,iconst,ric)._var_names(slab,dst);
            }

            IPrintFunc(pf) => pf._var_names(slab,dst),

            IFuncIf{cond, then, els} => {
                let mut iconst : Instruction;
                ic_to_instr!(slab.cs,iconst,cond)._var_names(slab,dst);
                ic_to_instr!(slab.cs,iconst,then)._var_names(slab,dst);
                ic_to_instr!(slab.cs,iconst,els)._var_names(slab,dst);
            }
        }
    }
    fn eval(&self, slab:&Slab, ns:&mut impl EvalNamespace, tc:&impl TargetContext) -> Result<f64,Error> {
        match self {
            // I have manually ordered these match arms in a way that I feel should deliver good performance.
            // (I don't think this ordering actually affects the generated code, though.)

            IMul(li,ric) => {
                Ok( eval_compiled_ref!(get_instr!(slab.cs,li), slab, ns, tc) *
                    eval_ic_ref!(ric,slab,ns, tc) )
            }
            IAdd(li,ric) => {
                Ok( eval_compiled_ref!(get_instr!(slab.cs,li), slab, ns, tc) +
                    eval_ic_ref!(ric, slab, ns, tc) )
            }
            IExp{base, power} => {
                Ok( eval_ic_ref!(base, slab, ns, tc).powf(
                    eval_ic_ref!(power, slab, ns, tc) ) )
            }

            INeg(i) => Ok(-eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc)),
            IInv(i) => Ok(1.0/eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc)),

            IVar(name) => eval_var!(ns, name, Vec::new(), unsafe{ &mut *(std::ptr::addr_of!(slab.ps.char_buf) as *mut _) }),
            ITargetVar(name) => {
                match tc.lookup(name) {
                    Some(f) => Ok(f),
                    None => Err(Error::Undefined(name.to_string())),
                }
            }
            IFunc{name, args:ics} => {
                let mut args = Vec::with_capacity(ics.len());
                for ic in ics {
                    args.push( eval_ic_ref!(ic, slab, ns, tc) );
                }
                eval_var!(ns, name, args, unsafe{ &mut *(std::ptr::addr_of!(slab.ps.char_buf) as *mut _) })
            },

            IFuncIf {cond, then, els} => {
                let condition_result = eval_ic_ref!(cond, slab, ns, tc);
                if condition_result != 0.0 {
                    Ok(eval_ic_ref!(then, slab, ns, tc))
                } else {
                    Ok(eval_ic_ref!(els, slab, ns, tc))
                }
            }

            IFuncLog{base:baseic, of:ofic} => {
                let base = eval_ic_ref!(baseic, slab, ns, tc);
                let of = eval_ic_ref!(ofic, slab, ns, tc);
                Ok(log(base,of))
            }

            IFuncSin(i) => Ok( eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc).sin() ),
            IFuncCos(i) => Ok( eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc).cos() ),
            IFuncTan(i) => Ok( eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc).tan() ),
            IFuncASin(i) => Ok( eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc).asin() ),
            IFuncACos(i) => Ok( eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc).acos() ),
            IFuncATan(i) => Ok( eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc).atan() ),
            IFuncSinH(i) => Ok( eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc).sinh() ),
            IFuncCosH(i) => Ok( eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc).cosh() ),
            IFuncTanH(i) => Ok( eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc).tanh() ),
            IFuncASinH(i) => Ok( eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc).asinh() ),
            IFuncACosH(i) => Ok( eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc).acosh() ),
            IFuncATanH(i) => Ok( eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc).atanh() ),

            IFuncRound{modulus:modic, of:ofic} => {
                let modulus = eval_ic_ref!(modic, slab, ns, tc);
                let of = eval_ic_ref!(ofic, slab, ns, tc);
                Ok( (of/modulus).round() * modulus )
            }
            IMod{dividend, divisor} => {
                Ok( eval_ic_ref!(dividend, slab, ns, tc) %
                    eval_ic_ref!(divisor, slab, ns, tc) )
            }

            IFuncAbs(i) => Ok( eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc).abs() ),
            IFuncSign(i) => Ok( eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc).signum() ),
            IFuncInt(i) => Ok( eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc).trunc() ),
            IFuncCeil(i) => Ok( eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc).ceil() ),
            IFuncFloor(i) => Ok( eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc).floor() ),
            IFuncMin(li,ric) => {
                let left = eval_compiled_ref!(get_instr!(slab.cs,li), slab, ns, tc);
                let right = eval_ic_ref!(ric, slab, ns, tc);
                if left.is_nan() || right.is_nan() { return Ok(std::f64::NAN) }  // I need to implement NAN checks myself because the f64.min() function says that if one number is NaN, the other will be returned.
                if left<right {
                    Ok(left)
                } else {
                    Ok(right)
                }
            }
            IFuncMax(li,ric) => {
                let left = eval_compiled_ref!(get_instr!(slab.cs,li), slab, ns, tc);
                let right = eval_ic_ref!(ric, slab, ns, tc);
                if left.is_nan() || right.is_nan() { return Ok(std::f64::NAN) }
                if left>right {
                    Ok(left)
                } else {
                    Ok(right)
                }
            }


            IEQ(left, right) => {
                Ok( bool_to_f64!(f64_eq!(eval_ic_ref!(left, slab, ns, tc),
                                         eval_ic_ref!(right, slab, ns, tc))) )
            }
            INE(left, right) => {
                Ok( bool_to_f64!(f64_ne!(eval_ic_ref!(left, slab, ns, tc),
                                         eval_ic_ref!(right, slab, ns, tc))) )
            }
            ILT(left, right) => {
                Ok( bool_to_f64!(eval_ic_ref!(left, slab, ns, tc) <
                                 eval_ic_ref!(right, slab, ns, tc)) )
            }
            ILTE(left, right) => {
                Ok( bool_to_f64!(eval_ic_ref!(left, slab, ns, tc) <=
                                 eval_ic_ref!(right, slab, ns, tc)) )
            }
            IGTE(left, right) => {
                Ok( bool_to_f64!(eval_ic_ref!(left, slab, ns, tc) >=
                                 eval_ic_ref!(right, slab, ns, tc)) )
            }
            IGT(left, right) => {
                Ok( bool_to_f64!(eval_ic_ref!(left, slab, ns, tc) >
                                 eval_ic_ref!(right, slab, ns, tc)) )
            }

            INot(i) => Ok(bool_to_f64!(f64_eq!(eval_compiled_ref!(get_instr!(slab.cs,i), slab, ns, tc),0.0))),
            IAND(lefti, rightic) => {
                let left = eval_compiled_ref!(get_instr!(slab.cs,lefti), slab, ns, tc);
                if f64_eq!(left,0.0) { Ok(left) }
                else {
                    Ok(eval_ic_ref!(rightic, slab, ns, tc))
                }
            }
            IOR(lefti, rightic) => {
                let left = eval_compiled_ref!(get_instr!(slab.cs,lefti), slab, ns, tc);
                if f64_ne!(left,0.0) { Ok(left) }
                else {
                    Ok(eval_ic_ref!(rightic, slab, ns, tc))
                }
            }


            IPrintFunc(pf) => pf.eval(slab,ns, tc),


            // Put these last because you should be using the eval_compiled*!() macros to eliminate function calls.
            IConst(c) => Ok(*c),
            #[cfg(feature="unsafe-vars")]
            IUnsafeVar{ptr, ..} => unsafe { Ok(**ptr) },
        }
    }
}

