#include <tvm/ir/error.h>
#include <tvm/relay/adt.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/relay/analysis.h>

#include <stack>
#include <cassert>

namespace tvm {
namespace relay {

class iDomNode{
  public:
    int depth;
    Expr expr;
    iDomNode* parent;

    iDomNode(Expr _expr){
      depth = 1;
      expr = _expr;
      parent = nullptr;
    }
};

typedef std::unordered_map<Expr, iDomNode*, ObjectPtrHash, ObjectPtrEqual> UMAP;

iDomNode* _get_lca(iDomNode* lhs, iDomNode* rhs){
  while(lhs != rhs){
    if(!lhs || !rhs) return nullptr;

    if(lhs->depth < rhs->depth) rhs = rhs->parent;
    else if(lhs->depth > rhs->depth) lhs = lhs->parent;
    else{
      lhs = lhs->parent;
      rhs = rhs->parent;
    }
  }

  return lhs;
}

iDomNode* get_lca(Expr& expr, UMAP& memo){
  tvm::Array<Expr> inNodes;
  if(expr.as<CallNode>()){
    inNodes = static_cast<const CallNode*>(expr.get())->args;
  }else if(expr.as<TupleNode>()){
    inNodes = static_cast<const TupleNode*>(expr.get())->fields;
  }else if(expr.as<ConstantNode>() || expr.as<VarNode>()){
    // do nothing
  }else{
    assert(0 && "Not supported yet");
  }

  iDomNode* parent = nullptr;
  for(Expr inNode:inNodes){
      if(inNode.as<ConstantNode>() || inNode.as<VarNode>()){
        // do nothing
        continue;
      }

      if(parent) 
        parent = _get_lca(parent, memo[inNode]);
      else 
        parent = memo[inNode];
      
  }
  return parent;
}


void update_dom_node(Expr expr, UMAP& memo){
  assert(memo.count(expr));
  iDomNode* dnode = memo[expr];
  iDomNode* parent = get_lca(expr, memo);
  dnode->parent = parent;
  dnode->depth = (parent!=nullptr)?parent->depth+1:1;
}


TVM_REGISTER_GLOBAL("relay.analysis.dominance_analysis")
    .set_body_typed([](const Expr& expr, const Array<Expr>& node_list) {

        UMAP memo;
        int sz = node_list.size();
        for(int i=0;i<sz;i++){
          Expr _expr = node_list[i];
          memo[_expr] = new iDomNode(_expr);
        }

        assert(memo.size() == sz);
        
        for(int i=0;i<sz;i++){
          Expr _expr = node_list[i];
          //Expr _expr = node_list[sz-1-i];
          update_dom_node(_expr, memo);
        }

        tvm::Map<Expr, Expr> domTree;
        for(int i=0;i<sz;i++){
          Expr _expr = node_list[sz-1-i];
          auto _domNode = memo[_expr]->parent;
          //if(_expr.as<ConstantNode>() || _expr.as<VarNode>()) continue;

          if(_domNode) domTree.Set(_expr, _domNode->expr);
        }

        return domTree;
    });


}  // namespace relay
}  // namespace tvm
 
