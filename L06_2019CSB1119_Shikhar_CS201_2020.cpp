#include<bits/stdc++.h>
using namespace std;

#define mp make_pair
#define pb push_back
#define f(i,b) for(int (i)=0;(i)<(b);(i)++)
#define endl '\n'
#define fastio                        \
    ios_base::sync_with_stdio(false); \
    cin.tie(NULL);                    \
    cout.tie(NULL);
#define p1(x) x.first
#define p2(x) x.second

// General Purpose functions begin

void swap_int(int &x, int &y){
    int temp = x;
    x = y;
    y = temp;
}

void swap_pair(pair<int,int> &x, pair<int,int> &y){
    pair<int,int> temp = x;
    x = y;
    y = temp;
}

//General Purpose functions end

//Binary Heap Begins

void insert(vector<pair<int,int>> &heap, pair<int,int> input, int &k, vector<int> &pos){
    //w,v
    heap[k] = input;
    pos[p2(input)] = k;
    int s = k;
    while(s>0 && p1(heap[(s-1)/2]) > p1(heap[s])){
        //we insert at end and swap until parent has higher priority or it becomes highest priority
        swap_int(pos[p2(heap[(s-1)/2])],pos[p2(heap[s])]);
        swap_pair(heap[(s-1)/2],heap[s]);
        s = (s-1)/2;
    }
    ++k; //so now size has increased after insertion and making heap property valid again
}

pair<int,int> extract_min(vector<pair<int,int>> &heap, int &s, vector<int> &pos){
    pair<int,int> min = heap[0];
    
    swap_int(pos[p2(heap[0])],pos[p2(heap[s-1])]);
    swap_pair(heap[0], heap[s-1]);
    pos[p2(heap[s-1])] = -1;
    heap[s-1] = mp(0,0); //only for completeness and easy debugging
    s--;
    
    int l,r;
    int prev_small = -1;
    int small = 0;
    while(prev_small != small){
        //iterative heapify operation, makes sure the heap satisfies heap property
        prev_small = small;
        l = 2*small + 1;
        r = 2*small + 2;
        if(l < s && p1(heap[l]) < p1(heap[small])){
            small = l;
        }
        if(r < s && p1(heap[r]) < p1(heap[small])){
            small = r;
        }
        if(small != prev_small){
            swap_int(pos[p2(heap[prev_small])],pos[p2(heap[small])]);
            swap_pair(heap[prev_small], heap[small]);
        }
    }
    return min;
}

void decrease_key(vector<pair<int,int>> &heap, int v, int new_w, vector<int> &pos){
    int p = pos[v];
    if(p == -1){
        //for debugging
        cout << "Problem in find" << endl;
        return;
    }
    p1(heap[p]) = new_w;
    while(p > 0 && p1(heap[(p-1)/2]) > p1(heap[p])){
        //we swap until priority of parent is higher or it becomes the highest priority position
        swap_int(pos[p2(heap[(p-1)/2])],pos[p2(heap[p])]);
        swap_pair(heap[(p-1)/2],heap[p]);
        p = (p-1)/2;
    }
}

//Binary Heap ends

//Binomial Heap Begins

struct binheap{
    int w, v, deg;
    binheap *child, *par, *sib;
};

void check(binheap *heap){
    //for debugging
    while(heap != NULL){
        cout << heap->deg << " " << heap->w << " ";
        heap = heap->sib;
    }
}

binheap *JoinHeapB(binheap *heap1, binheap *heap2){
    //joins two heaps in ascending order while also maintaining ascending order property of the tree ranks/degree
    //works in a similar way to when we join sorted arrays in merge sort but with degree and linked lists
    
    binheap *temp;
    binheap *temp2;
    //we make the lowest degree element first
    if(heap1->deg <= heap2->deg){
        temp = heap1;
        temp2 = heap1;
        heap1 = heap1->sib;
    }
    else{
        temp = heap2;
        temp2 = heap2;
        heap2 = heap2->sib;
    }
    //keep connecting in asceding order of degree here
    while(heap1 && heap2){
        if(heap1->deg <= heap2->deg){
            temp2->sib = heap1;
            temp2 = heap1;
            heap1 = heap1->sib;
        }
        else{
            temp2->sib = heap2;
            temp2 = heap2;
            heap2 = heap2->sib;
        }
    }
    
    //we also make sure that the left over part is connected
    if(heap1 && !heap2){
        temp2->sib = heap1;
    }
    else if(heap2 && !heap1){
        temp2->sib = heap2;
    }
    
    return temp;
}

binheap *UnionB(binheap *heap1, binheap *heap2){
    if(!heap1 && !heap2){
        return NULL;
    }
    else if(!heap1){
        return heap2;
    }
    else if(!heap2){
        return heap1;
    }
    binheap *joined_heap = JoinHeapB(heap1, heap2);
    
    binheap *prev = NULL;
    binheap *curr = joined_heap;
    binheap *next = joined_heap->sib;

    //here we combine same degree trees and keep doing so while maintaining the fact that the
    //heap is arranged in ascending order of degree/rank of trees, that's why next->sib->deg condition
    while(1){
        if(curr->deg == next->deg && ((next->sib && next->deg != next->sib->deg) || !next->sib)){
            if(curr->w > next->w){
                //if prev was NULL then now the original heap starts with next
                //now the lowest degree in the heap and otherwise there isn't a problem

                if(!prev)
                    joined_heap = next;
                else
                    prev->sib = next;
                
                //making curr the child of next
                curr->sib = next->child;
                next->child = curr;
                curr->par = next;
                ++next->deg;
                
                curr = next;
            }
            else{
                curr->sib = next->sib;
                
                //making next the child of curr
                next->sib = curr->child;
                curr->child = next;
                next->par = curr;
                ++curr->deg;
            }
            next = curr->sib;
        }
        else{
            prev = curr;
            curr = curr->sib;
            next = next->sib;
        }
        if(!next){
            break;
        }
    }
    return joined_heap;
}

binheap *insertB(binheap *heap, int w, int v, map<int, binheap *> &bmap){
    //creates a node and does union on the heap
    binheap *temp = (binheap *)malloc(sizeof(binheap));
    temp->w = w;
    temp->v = v;
    temp->deg = 0;
    temp->child = temp->par = temp->sib = NULL;
    bmap[v] = temp;

    binheap *heap_final = UnionB(heap, temp);
    return heap_final;
}

binheap *extractminB(binheap *heap, int &v, int &w){
    //find-min, can be done in O(1) time, but extract-min is log(n) either way
    binheap *temp = heap->sib;
    binheap *min = heap; //pointer to min
    binheap *prev = heap;
    binheap *min_prev = NULL;
    if(heap->sib){ //if only one binomial-tree then no need to check
        //find_min in heap
        while(temp){
            if(temp->w < min->w){
                min = temp;
                min_prev = prev;
            }
            prev = temp;
            temp = temp->sib;
        }
    }
    else{
        heap = NULL;
    }
    if(!min_prev)
        heap = min->sib;
    else
        min_prev->sib = min->sib;

    v = min->v;
    w = min->w;
    
    if(!min->child){//0 degree tree was deleted and no further processing needed
        min->sib = NULL;
        free(min);
        return heap;
    }
    //reverse of the linked-list i.e., min->child
    prev = NULL; //reusing variable
    temp = min->child;
    binheap *next = NULL;

    while(temp){
        temp->par = NULL; //making the parent NULL to ensure no data leakage
        next = temp->sib;
        temp->sib = prev;
        prev = temp;
        temp = next;
    }
    //prev is the small binomial heap of children of min node
    min->child = NULL;
    min->sib = NULL;
    free(min);

    return UnionB(heap, prev);
}

void updatekeyB(map<int, binheap *> &bmap, int v, int new_w){ //decrease key operation
    binheap *update = bmap[v]; //node to be updated

    update->w = new_w;
    binheap *par = update->par;
    binheap *temp;

    while(par && new_w < par->w){
        //all the data i.e., vertex label, priority label and map position is swapped
        //as long as the parent has lower priority value then update
        swap_int(par->w, update->w);

        temp = bmap[update->v];
        bmap[update->v] = bmap[par->v];
        bmap[par->v] = temp;

        swap_int(par->v, update->v);

        update = par;
        par = par->par;
    }
}

//Binomial Heap ends

//Fibonacci Heap begins

struct fibheap{
    int w, v, deg;
    bool col;
    fibheap *par, *child, *l, *r;
};

fibheap *insertF(fibheap *heap, int w, int v, vector<fibheap *> &pos){
    //joins elements in a circular doubly linked list, after making that one element

    fibheap *temp = (fibheap *)malloc(sizeof(fibheap));
    fibheap *temp2;
    temp->w = w;
    temp->v = v;
    temp->deg = 0;
    temp->par = NULL;
    temp->child = NULL;
    temp->l = temp;
    temp->r = temp;
    temp->col = false;
    pos[v] = temp;

    if(!heap){
        //start condition
        heap = temp;
    }
    else{
        temp2 = heap->l;
        temp2->r = temp;
        heap->l = temp;
        temp->l = temp2;
        temp->r = heap;
    }
    if(heap->w > w)
        heap = temp;
    return heap;
}

void display(fibheap *heap){
    //for debugging purpose
    fibheap *temp = heap;
    do{
        cout << temp->w << " " << temp->deg << " check" << endl;
        temp = temp->r;
    }while(temp != heap);
}

fibheap *joinheapF(fibheap *heap1, fibheap *heap2){
    //merges two heaps in O(1)
    //this is similar to joining circular doubly linked list
    //it also makes sure that the minimum of the two is being pointed
    fibheap *mheap, *Mheap, *t1, *t2;
    mheap = (heap1->w > heap2->w)?(heap2):(heap1);
    Mheap = (heap1->w <= heap2->w)?(heap2):(heap1);
    
    t1 = mheap->r;
    mheap->r = Mheap;
    t2 = Mheap->l;
    Mheap->l = mheap;//connecting both heaps
    t1->l = t2;
    t2->r = t1;

    return mheap;
}

fibheap *extractminF(fibheap *heap){

    if(heap->child){
        //removing colour as well as making parent NULL
        //if not done we may face problems in decrease key
        fibheap *t = heap->child;
        while(1){
            t->par = NULL;
            t->col = false;
            t = t->r;
            if(t == heap->child)
                break;
        }
    }

    if(heap->r != heap){
        //multiple tree case

        fibheap *temp = heap;
        heap->l->r = heap->r;
        heap->r->l = heap->l;
        
        if(heap->child)
            heap = joinheapF(heap->child, heap->r);
        else
            heap = heap->r;
        
        temp->child = NULL;
        temp->r = NULL;
        temp->l = NULL;
        free(temp);
    }
    else if(heap->r == heap && !heap->child){
        //single node case, no need for further processing

        heap->l = NULL;
        heap->r = NULL;
        free(heap);
        return NULL;
    }
    else{
        //single tree case

        fibheap *temp = heap;
        heap = heap->child;
        //heap may not point to the smallest elemenet
        //here and needs further processing 
        temp->child = NULL;
        temp->r = NULL;
        temp->l = NULL;
        free(temp);
    }

    //this is supposed to be the size of log2(total nodes in the heap) i.e., max of total = 10^7
    //this keeps track of the degree/rank of trees and and identifies same ranked trees as
    //in this unlike binomial heap we don't have the heap arranged by rank/degree

    vector<fibheap *> rank(100, NULL); //100 to be on the safe side, even 24 was fine

    //holder saves the same rank/degree tree encountered previously
    fibheap *holder = rank[heap->deg];

    while(holder != heap){
        
        if(holder){
            if(holder->w < heap->w){
                //in case a preivous tree of same degree exists and it's root element is
                //of higher priority then the current one i.e., heap then we detach it
                //make it's position the same as heap and detach heap and make it the child
                
                if(!(heap->r == holder && heap->l == holder)){//special case when there's two trees of same degree only
                    holder->l->r = holder->r;
                    holder->r->l = holder->l;
                    holder->l = heap->l;
                    holder->r = heap->r;
                }
                heap->l->r = holder;
                heap->r->l = holder;
                heap->par = holder;
                heap->r = heap;
                heap->l = heap;
                ++holder->deg;
                if(holder->child)//in case of multiple children we mere heap with them
                    holder->child = joinheapF(holder->child, heap);
                else
                    holder->child = heap;
                heap = holder; //changing heap to holder
            }
            else{
                //similar to above condition
                holder->l->r = holder->r;
                holder->r->l = holder->l;
                //disconnecting holder from the main chain
                holder->l = holder->r = holder;
                holder->par = heap;
                ++heap->deg;
                if(heap->child)
                    heap->child = joinheapF(heap->child, holder);
                else
                    heap->child = holder;
            }
        }
        else{
            //in case if that degree tree wasn't found then we save it and move to next tree
            rank[heap->deg] = heap;
            heap = heap->r;
        }
        //finding same ranked tree and storing in holder
        holder = rank[heap->deg];
        rank[heap->deg] = NULL;
    }

    //finding the minimum again, it's important to do it after all this
    //otherwise there maybe problems in special cases with same priority
    //being parent and children and being the minimum too, it takes log(N)
    //time at maximum and doesn't affect the time complexity of extract min

    fibheap *spec = heap->r;
    fibheap *pnt = heap;
    while(spec != pnt){
        if(spec->w < heap->w){
            heap = spec;
        }
        spec = spec->r;
    }
    return heap;
}

fibheap *updatekeyF(fibheap *heap, int v, int new_w, vector<fibheap *> pos){
    fibheap *update = pos[v];
    update->w = new_w;
    fibheap *curr_par;
    
    if(!update->par || update->par->w <= new_w){
        //if the parent is lesser or update is a root we return
        if(heap->w > new_w)
            heap = update;
        return heap;
    }
    
    do{
        update->l->r = update->r;
        update->r->l = update->l;
        update->par->child = (update->r != update)?(update->r):(NULL);
        update->r = update->l = update; //a small heap with a single tree is created with update as minimum
        //unmarking, if coloured
        update->col = false;
        --update->par->deg;
        curr_par = update->par;
        update->par = NULL;
        //merging both of the heaps, basically connecting two circular doubly linked lists
        heap = joinheapF(heap, update);
        
        update = curr_par;
        
        //we check if update's parent is coloured and repeat above if marked
    }while(update->col && update->par);
    
    if(update->par){
        //update's parent is made to be coloured for further decrease key operations
        //unless it's the root of a tree, in that case there's no point in colouring it
       update->col = true; 
    }
    return heap;
}

//Fibonacci Heap ends

//Johnson's Algo Code begins

void Dijkstra_FIBONACCI_HEAP(vector<vector<int>> v, vector<vector<int>> w, int s, int n, vector<vector<int>> &shortest_path){
    fibheap *heap = NULL;
    vector<fibheap *> pos(n+1, NULL);

    f(i,n){
        shortest_path[s][i+1] = 999999;
    }

    shortest_path[s][s] = 0;
    heap = insertF(heap, 0, s, pos);
    
    while(heap){
        int l = heap->v;
        pos[heap->v] = NULL;
        heap = extractminF(heap);

        f(i,v[l].size()){
            if(shortest_path[s][v[l][i]] > shortest_path[s][l] + w[l][i]){
               if(shortest_path[s][v[l][i]] != 999999){
                   //decrease_key
                   heap = updatekeyF(heap, v[l][i], shortest_path[s][l] + w[l][i], pos);
               }
               else{
                   //if not present in heap already insert
                   heap = insertF(heap, shortest_path[s][l] + w[l][i], v[l][i], pos);
               }
               shortest_path[s][v[l][i]] = shortest_path[s][l] + w[l][i];
            }
        }
    }
}

void Dijkstra_BINOMIAL_HEAP(vector<vector<int>> v, vector<vector<int>> w, int s, int n, vector<vector<int>> &shortest_path){
    binheap *bheap = NULL;
    map<int, binheap *> bmap;//stores the address for access only, can be done using vectors too

    f(i,n){
        shortest_path[s][i+1] = 999999;
    }

    shortest_path[s][s] = 0;
    bheap = insertB(bheap, 0, s, bmap);
    
    while(bheap){
        int l,lw;
        bheap = extractminB(bheap, l, lw);
        f(i,v[l].size()){
            if(shortest_path[s][v[l][i]] > shortest_path[s][l] + w[l][i]){
               if(shortest_path[s][v[l][i]] != 999999){
                   //decrease_key
                   updatekeyB(bmap, v[l][i], shortest_path[s][l] + w[l][i]);
               }
               else{
                   //if not present in heap already insert
                   bheap = insertB(bheap, shortest_path[s][l] + w[l][i], v[l][i], bmap);
               }
               shortest_path[s][v[l][i]] = shortest_path[s][l] + w[l][i];
            }
        }
    }
}

void Dijkstra_BINARY_HEAP(vector<vector<int>> v, vector<vector<int>> w, int s, int n, vector<vector<int>> &shortest_path){
    vector<pair<int,int>> heap(n); //using this as an array
    vector<int> pos(n+1,-1); //helps make find operation for decrease key O(1)

    f(i,n){
        shortest_path[s][i+1] = 999999;
    }
    shortest_path[s][s] = 0;
    int size = 0;
    insert(heap, mp(0,s), size, pos);
    while(size){
        pair<int,int> x = extract_min(heap, size, pos);

        f(i,v[p2(x)].size()){
            if(shortest_path[s][v[p2(x)][i]] > shortest_path[s][p2(x)] + w[p2(x)][i]){
               if(pos[v[p2(x)][i]] != -1){
                   //decrease_key
                    decrease_key(heap, v[p2(x)][i], shortest_path[s][p2(x)] + w[p2(x)][i], pos);
               }
               else{
                   //if not present in heap already insert
                   insert(heap, mp(shortest_path[s][p2(x)] + w[p2(x)][i], v[p2(x)][i]), size, pos);
               }
               shortest_path[s][v[p2(x)][i]] = shortest_path[s][p2(x)] + w[p2(x)][i];
            }
        }
    }
}

void Dijkstra_ARRAY(vector<vector<int>> v, vector<vector<int>> w, int s, int n, vector<vector<int>> &shortest_path){

    //O(M + N^2) for the array version

    vector<int> done(n+1,0);
    f(i,n){ //using this as an array
        shortest_path[s][i+1] = 999999;
    }
    shortest_path[s][s] = 0;
    int size = 1;
    while(size){
        int l;
        int min = 999999;
        f(i,n){
            //getting min by traversing the array i.e., O(n) as opposed to other heaps with O(log(n))
            if(!done[i+1] && shortest_path[s][i+1] < min){
                l = i+1; //min vertex
                min = shortest_path[s][i+1];
            }
        }
        done[l] = 1;
        size--;

        f(i,v[l].size()){
            if(shortest_path[s][v[l][i]] > shortest_path[s][l] + w[l][i]){
                if(shortest_path[s][v[l][i]] == 999999){
                    size++;
                }
               shortest_path[s][v[l][i]] = shortest_path[s][l] + w[l][i];
            }
        }
    }
}

int BellmanFord(vector<vector<int>> v, vector<vector<int>> &w, int n, vector<int> &dis){
    //handles adding dummy vertex and changes the graph edge weights
    f(i,n){//dummy vertex
        v[0].pb(i+1);
        w[0].pb(0);
    }

    dis[0] = 0;

    int to_break;
    f(i,n+1){
        to_break = 0;
        f(j,n+1){
            //j is u
            f(k,v[j].size()){
                //(v[j][k]) is v
                if(dis[j] != 999999 && dis[v[j][k]] > dis[j] + w[j][k]){
                    if(i == n){
                        //negative cycle found
                        return 1;
                    }
                    to_break = 1;
                    dis[v[j][k]] = dis[j] + w[j][k];
                }
            }
        }
        if(!to_break) //no updates, no need to run bellman ford further 
            break;
    }
    
    //updating weights such that now they are all positive and Dijkstra can work on them
    f(i,n){
        f(j,v[i+1].size()){
            w[i+1][j] = w[i+1][j] + dis[i+1] - dis[v[i+1][j]];
        }
    }
    return 0;
}

void JohnsonAlgo(vector<vector<int>> v, vector<vector<int>> w, int n, int d, int ch){

    vector<int> dis(n+1, 999999);
    int confirm_neg_cycle = BellmanFord(v, w, n, dis);
    if(confirm_neg_cycle){
        cout << -1 << endl;
        return;
    }
    //now we apply dijkstra to calculate distance for each vertex
    vector<vector<int>> shortest_path(n+1, vector<int>(n+1));
    f(i,n){
        //i+1 is the source here at each iteration
        if(ch == 1){
            Dijkstra_ARRAY(v, w, i+1, n, shortest_path);
        }
        else if(ch == 2){
            Dijkstra_BINARY_HEAP(v, w, i+1, n, shortest_path);
        }
        else if(ch == 3){
            Dijkstra_BINOMIAL_HEAP(v, w, i+1, n, shortest_path);
        }
        else{
            Dijkstra_FIBONACCI_HEAP(v, w, i+1, n, shortest_path);
        }
    }

    f(i,n){
        f(j,n){
            //i+1 to j+1
            if(shortest_path[i+1][j+1] != 999999)
                shortest_path[i+1][j+1] += dis[j+1] - dis[i+1];
            cout << shortest_path[i+1][j+1] << " ";
        }
        cout << endl;
    }
}

//Johnson's Algo Code ends

int main(int argc, char** argv) 
{ 
    fastio
	int t,n,d,ch;
    if(argc == 1 || argv[argc-1][0] == '4'){
        ch = 4;
    }
    else if(argv[argc-1][0] == '3'){
        ch = 3;
    }
    else if(argv[argc-1][0] == '2'){
        ch = 2;
    }
    else{
        ch = 1;
    }
    vector<double> time_t;
    cin >> t;
    f(zz,t){
        clock_t start, end;
        start = clock();
        cin >> n >> d;
        vector<vector<int>> v(n+1);
        vector<vector<int>> w(n+1);
        int x;
        f(i,n){
            f(j,n){
                cin >> x;
                if(i != j && x != 999999){
                    v[i+1].pb(j+1);
                    w[i+1].pb(x);
                }
            }
        }
        JohnsonAlgo(v, w, n, d, ch);
        end = clock();
        time_t.pb(((double) (end - start)) / double(CLOCKS_PER_SEC));
    }
    for(vector<double>::iterator i = time_t.begin(); i != time_t.end(); i++){
        cout << (*i) << " ";
    }
} 
/*I have checked my code against multiple inputs and described it in the report I have submitted. The code worked well
on all inputs. I have run small inputs of upto 60 nodes on online compilers but for more than that I have used my laptop.
*/