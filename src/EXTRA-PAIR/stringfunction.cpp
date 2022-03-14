#include <string.h>
#include <assert.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "stringfunction.h"

static double min_func(double a,double b) { return (a < b) ? a:b; }
static double max_func(double a,double b) { return (a > b) ? a:b; }

static double add_func(double a,double b) { return a+b; }
static double sub_func(double a,double b) { return a-b; }
static double mul_func(double a,double b) { return a*b; }
static double div_func(double a,double b) { return a/b; }
static double neg_func(double a) { return -a; }


static void dadd_func(double a,double b,double da[1],double db[1]) { da[0] = 1; db[0] =  1; }
static void dsub_func(double a,double b,double da[1],double db[1]) { da[0] = 1; db[0] = -1; }
static void dmul_func(double a,double b,double da[1],double db[1]) { da[0] = b; db[0] =  a; }
static void ddiv_func(double a,double b,double da[1],double db[1]) { da[0] = 1/b; db[0] = -a/(b*b); }
static double dneg_func(double a) { return -1; }



static double sgn_func(double x) {
  if(x < 0.0) return -1.0;
  if(x >= 0.0) return 1.0;
  return x; /* x is NaN */
}
static double dcos_func(double x) { return -sin(x); }
static double dtan_func(double x) {
  const double c = cos(x);
  return 1.0 + c*c;
}
static double inv_func(double x) { return 1.0/x; }

static void dmax_func(double a,double b,double da[1],double db[1]) {
  double s = 0.0;
  if(a >= b) s = 1.0;
  da[0] = s;
  db[0] = 1.0-s;
}
static void dmin_func(double a,double b,double da[1],double db[1]) {
  double s = 0.0;
  if(a <= b) s = 1.0;
  da[0] = s;
  db[0] = 1.0-s;
}
static void dpow_func(double a,double b,double da[1],double db[1]) {
  /* a^b = exp(b*log(a));
     d/da (a^b) = a^(b-1) * b;
     d/db (a^b) = a^b * log(a);
  */
  /*
  const double
    alog = log(a),
    abm1 = exp((b-1.0)*alog);
  da[0] = abm1 * b;
  db[0] = abm1 * a * alog;
  */
  const double atobm1 = (b == 1) ? 1 : pow(a,b-1);
  da[0] = b*atobm1;
  if(a == 0)
    db[0] = 0;
  else
    db[0] = log(fabs(a))*atobm1*a;
}

static struct {
  const char *name;
  int narg;
  funt func,dfunc;
} func_table[] = {
  { "abs" , 1 , fabs , sgn_func  } ,
  { "sin" , 1 , sin  , cos       } ,
  { "cos" , 1 , cos  , dcos_func } ,
  { "tan" , 1 , tan  , dtan_func } ,
  { "exp" , 1 , exp  , exp       } ,
  { "log" , 1 , log  , inv_func  }/* ,
  { "pow" , 2 , (funt) pow  , (funt) dpow_func } ,
  { "min" , 2 , (funt) static_cast<double (*)(double,double)>(min_func) , (funt) dmax_func } ,
  { "max" , 2 , (funt) max_func , (funt) dmin_func }*/
};
static const int nfuncs = sizeof(func_table)/sizeof(*func_table);

static node *make_node(int type,int narg,funt func,funt dfunc,node *arg1,node *arg2) {
  node *n = (node *) calloc(1,sizeof(node));
  n->type = type;
  n->narg = narg;
  n->func = func;
  n->dfunc = dfunc;
  n->eval_tag = 0;
  n->args[0] = arg1;
  n->args[1] = arg2;
  return n;
}


static void clear_tree(node *tree) {
  if(tree != nullptr) {
    int i;
    tree->eval_tag = 0;
    for(i = 0; i<tree->narg; i++)
      clear_tree(tree->args[i]);
  }
}
static void ref_count(node *tree) {
  if(tree != nullptr) {
    int i;
    tree->eval_tag++;
    for(i = 0; i<tree->narg; i++)
      ref_count(tree->args[i]);
  }
}
static void delete_tree_core(node *tree) {
  if(tree != nullptr) {
    tree->eval_tag--;
    if(tree->eval_tag == 0) {
      int i;
      for(i = 0; i<tree->narg; i++)
	delete_tree_core(tree->args[i]);
      free(tree);
    }
  }
}
void delete_tree(node *tree) {
  if(tree != nullptr) {
    if(tree->type == LINEAR_TREE)
      free(tree);
    else {
      clear_tree(tree);
      ref_count(tree);
      delete_tree_core(tree);
    }
  }
} 


/* Tree reduction (common sub-expression elimination) and evaluation */
typedef struct {
  int nnodes,nalloc;
  node **nodelist;
} opt_nodelist;

static node *reduce_tree_core(node *tree,opt_nodelist *onp) {
  if(tree != nullptr) {
    int i;
    for(i = 0; i<tree->narg; i++) {
      node *nptr = reduce_tree_core(tree->args[i],onp);
      if(nptr != tree->args[i]) {
	free(tree->args[i]);
	tree->args[i] = nptr;
      }
    }
    if(tree->type == COMMUTATIVE) {
      if(tree->args[0] - tree->args[1] > 0) {
	node *nptr = tree->args[0];
	tree->args[0] = tree->args[1];
	tree->args[1] = nptr;
      }
    }
    for(i = 0; i<onp->nnodes; i++)
      if(memcmp(tree,onp->nodelist[i],sizeof(node)) == 0)
	return onp->nodelist[i];
    if(onp->nnodes >= onp->nalloc) {
      onp->nalloc *= 2;
      onp->nodelist = (node **) realloc(onp->nodelist,sizeof(node *) * onp->nalloc);
    }
    onp->nodelist[onp->nnodes++] = tree;
  }
  return tree;
}

static void reduce_tree(node *tree) {
  node *ntree;
  opt_nodelist opt_data = {0,10,nullptr};
  opt_data.nodelist = (node **) malloc(sizeof(node *) * opt_data.nalloc);
  
  ntree = reduce_tree_core(tree,&opt_data);
  free(opt_data.nodelist);
  if(ntree != tree) {
    fprintf(stderr,
	    "%s:%d in %s(): Crap, reduce_tree_core should return root node...\n",
	    __FILE__,__LINE__,__func__);
    exit(1);
  }
}

static void eval_tree_core(node *tree,int nparms,double parms[]) {
  if(tree != nullptr) {
    int i;
    if(tree->eval_tag > 0) return;
    for(i = 0; i<tree->narg; i++)
      eval_tree_core(tree->args[i],nparms,parms);
    switch(tree->type) {
    case CONSTANT:
      break;
    case PARAMETER:
      tree->val = parms[-tree->narg];
      break;
    case ONEFUNCTION:
      tree->val = ((unary_func) tree->func) (tree->args[0]->val);
      break;
    case TWOFUNCTION:
    case COMMUTATIVE:
      tree->val = ((binary_func) tree->func) (tree->args[0]->val,tree->args[1]->val);
      break;
    default:
      fprintf(stderr,
	      "%s:%d in %s(): Crap, undefined node type %d...\n",
	      __FILE__,__LINE__,__func__,tree->type);
      exit(1);
    }
    tree->eval_tag = 1;
  }
}

double eval_tree(node *tree,int nparms,double parms[]) {
  clear_tree(tree);
  eval_tree_core(tree,nparms,parms);
  return tree->val;
}


static void eval_tree_core_deriv(node *tree,int nparms,double parms[][2]) {
  if(tree != nullptr) {
    int i;
    if(tree->eval_tag > 0) return;
    for(i = 0; i<tree->narg; i++)
      eval_tree_core_deriv(tree->args[i],nparms,parms);
    switch(tree->type) {
    case CONSTANT:
      break;
    case PARAMETER:
      tree->val  = parms[-tree->narg][0];
      tree->dval = parms[-tree->narg][1];
      break;
    case ONEFUNCTION:
      {
	const double
	  a  = tree->args[0]->val,
	  da = tree->args[0]->dval;
	tree->val = ((unary_func) tree->func) (a);
	if(da == 0.0) tree->dval = 0.0;
	else tree->dval = ((unary_func) tree->dfunc) (a) * da;
      }
      break;
    case TWOFUNCTION:
    case COMMUTATIVE:
      {
	const double
	  a  = tree->args[0]->val,
	  da = tree->args[0]->dval,
	  b  = tree->args[1]->val,
	  db = tree->args[1]->dval;
	tree->val = ((binary_func) tree->func) (a,b);
	if(da == 0.0 && db == 0.0)
	  tree->dval = 0.0;
	else {
	  double dfda,dfdb;
	  ((two_in_two_out) tree->dfunc) (a,b,&dfda,&dfdb);
	  tree->dval = dfda*da + dfdb*db;
	}
      }
      break;
    default:
      fprintf(stderr,
	      "%s:%d in %s(): Crap, undefined node type %d...\n",
	      __FILE__,__LINE__,__func__,tree->type);
      exit(1);
    }
    tree->eval_tag = 1;
  }
}

double eval_tree_deriv(node *tree,int nparms,double parms[][2]) {
  clear_tree(tree);
  eval_tree_core_deriv(tree,nparms,parms);
  return tree->val;
}


/* Tree linearization and linear evaluator */

void tree_linearize_core(node *tree,node **optlist,int typeflag) {
  int i;
  assert(tree != nullptr);

  for(i = 0; i<tree->narg; i++)
    tree_linearize_core(tree->args[i],optlist,typeflag);

  if(tree->eval_tag > 0) return;
  if(typeflag >= 0 && tree->type != typeflag) return;
  if(tree->type == PARAMETER) {
    tree->eval_tag = -tree->narg + 1;
  } else {
    int nalloc = (*optlist)->eval_tag,nused = (*optlist)->narg;
    
    if(nused >= nalloc) {
      nalloc *= 2;
      *optlist = (node *) realloc(*optlist,sizeof(node) * nalloc);
      (*optlist)->eval_tag = nalloc;
    }
    
    /*
    printf("Adding node ptr=0x%8x type=%d narg=%d tag=%2d at position %2d, "
	   "args=0x%8x,0x%8x  typeflag = %2d\n",
	   (unsigned int) tree,tree->type,tree->narg,
	   tree->eval_tag,nused,
	   (unsigned int) tree->args[0],
	   (unsigned int) tree->args[1],
	   typeflag);
    */
    tree->eval_tag = nused;
    (*optlist)[nused] = *tree; /* Copy node to linear list */
    (*optlist)->narg = nused+1;
  }
}

static node * tree_linearize(node *tree,int nparm) {
  int nalloc = nparm + 10 + 1;
  node *optlist = (node *) malloc(sizeof(node) * nalloc);
  int i,firstnonconstant,lastnode,nused;

  optlist->type = LINEAR_TREE;
  optlist->narg = nparm+1;    /* Number of used nodes. */
  optlist->eval_tag = nalloc; /* Number of allocated nodes. (After list is fully populated,
				 the meaning of eval_tag of first node is redefined to the
				 index of the first non-constant node.) */

  for(i = 0; i<nparm; i++) {
    optlist[i+1].type = PARAMETER;
    optlist[i+1].narg = -i;
    optlist[i+1].func = nullptr;
  }
  clear_tree(tree);
  /* printf("Adding constants...\n"); */
  tree_linearize_core(tree,&optlist,CONSTANT);
  firstnonconstant = optlist->narg;
  /* printf("Adding everyting else..\n"); */
  tree_linearize_core(tree,&optlist,-1);

  /* Final size fixed. Redefine meaning of eval_tag
     to index first non constant node */
  optlist->eval_tag = firstnonconstant;
  nused = optlist->narg;

  lastnode = tree->eval_tag;
  assert(lastnode == nused-1);

  /* Let the arguments point directly to node values,
     instead of just the nodes to save pointer
     arithmetic at evaluation time. */
  for(i = firstnonconstant; i<nused; i++) {
    int j;
    for(j = 0; j<optlist[i].narg; j++) {
      const node *argptr = optlist[i].args[j];
      const int argidx = argptr->eval_tag;
      /*
      printf("  ##  node %d, argument %d = 0x%08x  idx = %d\n",
	     i,j,(unsigned int) argptr,argidx);
      */

      optlist[i].args[j] =
	(node *) &(optlist[argidx].val);
    }
  }
  return optlist;
}

double eval_optlist(node *optlist,int nparms,double parms[]) {
  const int nused = optlist->narg,firstnonconstant = optlist->eval_tag;
  int i;

  for(i = 0; i<nparms; i++)
    optlist[i+1].val = parms[i];

  for(i = firstnonconstant; i<nused; i++) {
    /* Only functions (operators) left in expression.
       constants and parameters are already evaluated,
       and are placed first in the list. */
    double x,a = *((double *) optlist[i].args[0]);
    binary_func func = reinterpret_cast<binary_func>(optlist[i].func);

    if(optlist[i].narg == 1) x = ((unary_func) func)(a);
    else /*if(optlist[i].narg == 2)*/ {
      double b = *((double *) optlist[i].args[1]);
      x = func(a,b);
    }
    optlist[i].val = x;
  }

  return optlist[nused-1].val;
}
double eval_optlist2(node *optlist,int nparms,double parms[]) {
  const int nused = optlist->narg,firstnonconstant = optlist->eval_tag;
  int i;

  for(i = 0; i<nparms; i++)
    optlist[i+1].val = parms[i];

  for(i = firstnonconstant; i<nused; i++) {
    /* Only functions (operators) left in expression.
       constants and parameters are already evaluated,
       and are placed first in the list. */

    double x,a = *((double *) optlist[i].args[0]);
    binary_func func = reinterpret_cast<binary_func>(optlist[i].func);
    double x11 = -a,x12 = fabs(a);

    if(optlist[i].narg == 1) {
      unary_func func1 = (unary_func) func;
      if(func1 == neg_func) x = x11;
      else if(func1 == static_cast<funt>(fabs)) x = x12;
      else x = func1(a);
    } else /*if(optlist[i].narg == 2)*/ {
      double b = *((double *) optlist[i].args[1]);
      double x1 = a+b,x2 = a-b,x3 = a*b;

      if     (func == add_func) x = x1;
      else if(func == sub_func) x = x2;
      else if(func == mul_func) x = x3;
      else if(func == div_func) x = a/b;
      else x = func(a,b);

    }
    optlist[i].val = x;
  }

  return optlist[nused-1].val;
}

/* Print linearized tree */
void optlist_print(node *optlist) {
  const int nused = optlist->narg,firstnonconstant = optlist->eval_tag;
  int i;

  printf("nused = %d, firstnonconstant = %d\n",nused,firstnonconstant);

  printf("Types:\n"
	 "  CONSTANT    = %d\n"
	 "  PARAMETER   = %d\n"
	 "  ONEFUNCTION = %d\n"
	 "  TWOFUNCTION = %d\n"
	 "  COMMUTATIVE = %d\n"
	 "  NTYPES      = %d\n",
	 CONSTANT,PARAMETER,ONEFUNCTION,TWOFUNCTION,COMMUTATIVE,NTYPES);

  for(i = 1; i<nused; i++) {
    node *off = (node *) &optlist->val;
    printf("Node %d:  type=%d  narg=%2d  value = %15.5e,  arg0 = %3d  arg1 = %3d\n",
	   i,optlist[i].type,optlist[i].narg,optlist[i].val,
	   (int) (optlist[i].args[0] - off),
	   (int) (optlist[i].args[1] - off));
  }
  printf("---\n");
}



/* Parsing functions */
static char *skipspace(char *s) {
  return s + strspn(s," \t\r\n");
}

typedef struct ERRLINK {
  const char *msg,*func,*file;
  int line;
  struct ERRLINK *next;
} errlink;

static errlink *make_error(const char *msg,
		    const char *func,
		    const char *file,int lineno,errlink *next) {
  errlink *err = (errlink *) malloc(sizeof(errlink));
  err->msg = msg;
  err->func = func;
  err->file = file;
  err->line = lineno;
  err->next = next;
  return err;
}

#define ret_w_err(msg)						  \
  do {								  \
    *errptr = make_error(msg,__func__,__FILE__,__LINE__,*errptr); \
    return nullptr;						  \
  } while(0);

static node *parse_expr(char **s,int nparms,char *parms[],errlink **errptr);

static node *parse_token(char **s,int nparms,char *parms[],errlink **errptr) {
  node *tok = nullptr;

  *s = skipspace(*s);
  if(**s == '(') {
    *s = *s + 1;
    tok = parse_expr(s,nparms,parms,errptr);
    if(tok == nullptr) ret_w_err("Invalid expression in prenthesis.");
    *s = skipspace(*s);
    if(**s != ')') {
      delete_tree(tok);
      ret_w_err("Missing right parenthesis.");
    }
    *s = *s + 1;
  } else if(isdigit(**s)) {
    tok = make_node(CONSTANT,-99,nullptr,nullptr,nullptr,nullptr);
    tok->val = strtod(*s,s);
  } else if(isalpha(**s)) {
    char *ptr = *s;
    while(isalnum(**s)) *s = *s + 1;
    if(**s == '(') {
      node *a1 = nullptr,*a2 = nullptr;
      int i;
      //printf("Parsing function, part is '%s' - '%s'.\n",ptr,*s);
      for(i = 0; i<nfuncs; i++) {
	//printf("COmparing with function %d of name %s\n",i,func_table[i].name);
	if(strncmp(ptr,func_table[i].name,*s-ptr) == 0)
	  break;
      }
      if(i >= nfuncs) ret_w_err("Unknown function name.");
      //printf("Found function %d...\n",i);
      *s = *s + 1;
      a1 = parse_expr(s,nparms,parms,errptr);
      if(a1 == nullptr) ret_w_err("Syntax error in first argument to function.");
      *s = skipspace(*s);
      if(**s == ',') {
	if(func_table[i].narg != 2) {
	  delete_tree(a1);
	  ret_w_err("Function does not take two arguments.");
	}
	*s = *s + 1;
	a2 = parse_expr(s,nparms,parms,errptr);
	if(a2 == nullptr) {
	  delete_tree(a1);
	  ret_w_err("Syntax error in second argument to function.");
	}
	*s = skipspace(*s);
      } else if(func_table[i].narg != 1) {
	  delete_tree(a1);
	  ret_w_err("Function does not take one argument.");
      }

      if(**s != ')') {
	delete_tree(a1);
	if(a2 != nullptr) {
	  delete_tree(a2);
	  ret_w_err("Missing right parenthesis after function argument(s)");
	}
      }
      *s = *s + 1;
      if(a2 == nullptr)
	tok = make_node(ONEFUNCTION,1,func_table[i].func,func_table[i].dfunc,a1,nullptr);
      else
	tok = make_node(TWOFUNCTION,2,func_table[i].func,func_table[i].dfunc,a1,a2);
    } else {
      int i;
      //printf("-- parsing parameter, parm is '%s' - '%s', legth %d.\n",ptr,*s,(int) (*s-ptr));
      for(i = 0; i<nparms; i++) {
	//printf("Checking with parameter %d name=%s\n",i,parms[i]);
	if(strncmp(ptr,parms[i],*s-ptr) == 0)
	  break;
      }
      if(i >= nparms) ret_w_err("Unknown parameter");
      /*
      printf("Found that token begining at '%s' parses to parameter %d with name %s.\n",
	     ptr,i,parms[i]);
      */
      tok = make_node(PARAMETER,-i,nullptr,nullptr,nullptr,nullptr);
    }
  }
  return tok;
}

static node *parse_factor(char **s,int nparms,char *parms[],errlink **errptr) {
  node *p = parse_token(s,nparms,parms,errptr);
  if(p == nullptr) ret_w_err("Syntax error in mantissa.");
  *s = skipspace(*s);
  while(**s == '^') {
    node *np;
    *s = *s + 1;
    np = parse_token(s,nparms,parms,errptr);
    if(np == nullptr) {
      delete_tree(p);
      ret_w_err("Syntax error in exponent.");
    }
    p = make_node(TWOFUNCTION,2,(funt) static_cast<double (*)(double,double)>(pow),(funt) dpow_func,p,np);
    *s = skipspace(*s);
  }
  return p;
}

static node *parse_term(char **s,int nparms,char *parms[],errlink **errptr) {
  node *fac = nullptr;

  fac = parse_factor(s,nparms,parms,errptr);
  if(fac == nullptr) ret_w_err("Syntax error in factor.");
  *s = skipspace(*s);
  while(**s == '*' || **s == '/') {
    node *nfac;
    char op = **s;
    *s = *s + 1;
    nfac = parse_factor(s,nparms,parms,errptr);
    if(nfac == nullptr) ret_w_err("Syntax error in factor.");
    if(op == '*')
      fac = make_node(COMMUTATIVE,2,(funt) mul_func,(funt) dmul_func,fac,nfac);
    else
      fac = make_node(TWOFUNCTION,2,(funt) div_func,(funt) ddiv_func,fac,nfac);
    *s = skipspace(*s);
  }
  return fac;
}

static node *parse_expr(char **s,int nparms,char *parms[],errlink **errptr) {
  node *tree = nullptr;

  while(*(*s = skipspace(*s)) != '\0') {
    int sign = 1;
    node *t;

    if(**s == '+' || **s == '-') {
      if(**s == '-') sign = -1;
      *s = skipspace(*s + 1);
    } else if(tree != nullptr) return tree;

    t = parse_term(s,nparms,parms,errptr);
    if(t == nullptr) {
      delete_tree(tree);
      ret_w_err("Syntax error in term.");
    }

    if(tree == nullptr) {
      if(sign == 1) tree = t;
      else tree = make_node(ONEFUNCTION,1,reinterpret_cast<funt>(neg_func),(funt) dneg_func,t,nullptr);
    } else {
      if(sign == 1)
	tree = make_node(COMMUTATIVE,2,reinterpret_cast<funt>(add_func),(funt) dadd_func,tree,t);
      else
	tree = make_node(TWOFUNCTION,2,reinterpret_cast<funt>(sub_func),(funt) dsub_func,tree,t);
    }
  }
  return tree;
}

node *parse_string(char *s,int nparms,char *parms[]) {
  errlink *errlist = nullptr;
  char *s0 = s;
  node *tree;

  if(0) {
    int i;
    printf("In parse string, %d parms, with names:\n",nparms);
    for(i = 0; i<nparms; i++)
      printf("  %2d = '%s'\n",i,parms[i]);
    printf("--\n");
  }

  tree = parse_expr(&s,nparms,parms,&errlist);

  if(tree != nullptr) {
    s = skipspace(s);
    if(*s != '\0' && *s != ';') {
      errlist = make_error("Unexpexted characters after end of expression",
			   __func__,__FILE__,__LINE__,errlist);
      delete_tree(tree);
      tree = nullptr;
    }
  }

  if(tree == nullptr) {
    errlink *p;
    fprintf(stderr,
	    "Error while parsing expression \'%s\'.\n"
	    "String left to parse at error detection \'%s\'.\n"
	    "Error messages:\n",s0,s);
    while(errlist != nullptr) {
      fprintf(stderr,
	      "  @ %s:%d in %s(): %s\n",
	      errlist->file,errlist->line,errlist->func,errlist->msg);
      p = errlist->next;
      free(errlist);
      errlist = p;
    }
    fprintf(stderr,"-- End of error messages --\n");
  } else {
    errlink *p;
    //reduce_tree(tree);
    while(errlist != nullptr) {
      p = errlist->next;
      free(errlist);
      errlist = p;
    }
  }
  return tree;
}
